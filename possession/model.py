"""Interpretable observation evidence and an exact offline three-state decoder."""

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.special import expit, softmax

from games import GAME, GAMES
from possession.contracts import positive_fps, validate_ranges
from possession.features import smooth


@dataclass(frozen=True)
class Config:
    # Scores integrate over seconds, making transition costs independent of sampling rate.
    control_weight: float = 3.0
    tag_weight: float = 1.8
    evidence_window_s: float = 0.6
    activity_window_s: float = 1.0
    live_bias: float = -1.0
    live_tag_weight: float = 2.6
    live_motion_weight: float = 2.0
    live_contact_weight: float = 1.0
    live_team_motion_weight: float = 2.2
    still_weight: float = 2.0
    outside_weight: float = 2.5
    restart_still_weight: float = 1.4
    gap_weight: float = 0.35
    team_switch_cost: float = 0.9
    dead_switch_cost: float = 1.2
    use_tags: bool = True
    use_dead_model: bool = True
    use_published_pregame: bool = True
    adaptive_transitions: bool = False
    motion_control: bool = False
    reliability_gated_tags: bool = False
    normalise_control: bool = False
    transition_discount: float = 0.75
    transition_context_s: float = 0.4

    def validate(self):
        boolean_fields = {
            "use_tags",
            "use_dead_model",
            "use_published_pregame",
            "adaptive_transitions",
            "motion_control",
            "reliability_gated_tags",
            "normalise_control",
        }
        for key, value in asdict(self).items():
            if key in boolean_fields:
                if type(value) is not bool:
                    raise ValueError(f"{key} must be boolean")
                continue
            if type(value) not in (int, float) or not np.isfinite(value):
                raise ValueError(f"{key} must be a finite number")
            if key != "live_bias" and value < 0:
                raise ValueError(f"{key} must be non-negative")
        if self.transition_discount > 1:
            raise ValueError("transition_discount must be between zero and one")
        return self

    @classmethod
    def read(cls, path=None):
        return cls(**json.loads(Path(path).read_text())).validate() if path else cls().validate()


def evidence(features, config, ranges, fps=25):
    """Return uncalibrated state log-scores and inspectable component signals."""
    config.validate()
    n = len(features["control"])

    def mean(key):
        return smooth(features[key], config.activity_window_s, fps, ranges)

    control_feature = "motion_control" if config.motion_control else "control"
    control = smooth(features[control_feature], config.evidence_window_s, fps, ranges)
    coverage = np.clip(
        smooth(features["ball_reliable"], config.evidence_window_s, fps, ranges), 0, 1
    )
    if config.normalise_control:
        # Missing observations are unknown, not evidence that control moved towards 50/50.
        # The floor caps amplification at 4x when only a few samples survive in the window.
        control = np.clip(control / np.maximum(coverage, 0.25), -1, 1)
    tag_team = features["tag_home"] - features["tag_away"]
    tag_live = features["tag_live"]
    set_piece = features["tag_set_piece"]
    if not config.use_tags:
        tag_team, tag_live, set_piece = np.zeros((3, n))
    quality = np.where(
        features["homography_valid"], 1 / (1 + np.maximum(features["homography_error"] - 1, 0)), 0
    )
    quality = np.nan_to_num(quality, nan=0)
    tag_gate = 1 - coverage * quality if config.reliability_gated_tags else np.ones(n)
    team_logit = config.control_weight * control * quality + config.tag_weight * tag_team * tag_gate
    motion = mean("ball_moving")
    still = mean("ball_still")
    team_motion = mean("moving_fraction")
    static_evidence = still * (1 - np.clip(team_motion * 2, 0, 1))
    live_logit = (
        config.live_bias
        + config.live_tag_weight * tag_live
        + config.live_motion_weight * motion
        + config.live_contact_weight * mean("contact")
        + config.live_team_motion_weight * team_motion
        - config.still_weight * static_evidence
        - config.outside_weight * mean("ball_outside")
        - config.restart_still_weight * set_piece * (1 - motion)
        - config.gap_weight * np.minimum(features["missing_age"] / 5, 1) * (1 - tag_live)
    )
    if not config.use_dead_model:
        # Deliberate ablation: missing ball alone acts as a weak stoppage cue.
        live_logit = 4 * mean("ball_reliable") - 1.5
    p_live = np.clip(expit(live_logit), 1e-6, 1 - 1e-6)
    p_home = np.clip(expit(team_logit), 1e-6, 1 - 1e-6)
    # Center team evidence at 0: uncertainty about team must not itself mean dead ball.
    scores = np.stack(
        [
            np.log(p_live) + np.log(2 * p_home),
            np.log(p_live) + np.log(2 * (1 - p_home)),
            np.log1p(-p_live),
        ],
        axis=1,
    )
    diagnostics = {
        "live_evidence": p_live,
        "home_evidence": p_home,
        "observation_confidence": softmax(scores, axis=1).max(axis=1),
        "control_coverage": coverage,
        "team_tag_gate": tag_gate if config.use_tags else np.zeros(n),
    }
    return scores, diagnostics


def viterbi(scores, transition_cost, fps=25):
    """Maximise integrated observation scores minus change costs; O(T K²), exact traceback."""
    scores = np.asarray(scores, dtype=np.float64)
    cost = np.asarray(transition_cost, dtype=np.float64)
    positive_fps(fps)
    if scores.ndim != 2 or not 1 <= scores.shape[1] <= 127:
        raise ValueError("Expected scores[T,K] with 1 <= K <= 127")
    shape = (scores.shape[1], scores.shape[1])
    if cost.shape not in (shape, (max(0, len(scores) - 1), *shape)):
        raise ValueError("Expected transition_cost[K,K] or transition_cost[T-1,K,K]")
    # -inf encodes a genuinely forbidden state, rather than an arbitrary finite penalty.
    if not (np.isfinite(scores) | np.isneginf(scores)).all():
        raise ValueError("Observation scores must be finite or negative infinity")
    if not np.isfinite(scores).any(axis=1).all():
        raise ValueError("At least one state must be allowed in every frame")
    if not np.isfinite(cost).all() or (cost < 0).any():
        raise ValueError("Transition costs must be finite and non-negative")
    if not len(scores):
        return np.array([], dtype=np.int8)
    k = scores.shape[1]
    back = np.zeros((len(scores), k), dtype=np.int8)
    value = scores[0] / fps
    state_index = np.arange(k)
    for t in range(1, len(scores)):
        candidates = value[:, None] - (cost if cost.ndim == 2 else cost[t - 1])
        back[t] = candidates.argmax(axis=0)
        value = candidates[back[t], state_index] + scores[t] / fps
        value -= value.max()
    states = np.empty(len(scores), dtype=np.int8)
    states[-1] = value.argmax()
    for t in range(len(scores) - 1, 0, -1):
        states[t - 1] = back[t, states[t]]
    return states


def transition_costs(control, config, fps=25):
    """Reduce change cost only at sustained, directional evidence of a control hand-over.

    Left/right windows are disjoint at the candidate boundary. No label or possession duration
    from ground truth is involved. Unknown control is zero and cannot create a discount.
    """
    base = np.array(
        [
            [0, config.team_switch_cost, config.dead_switch_cost],
            [config.team_switch_cost, 0, config.dead_switch_cost],
            [config.dead_switch_cost, config.dead_switch_cost, 0],
        ],
        dtype=float,
    )
    if not config.adaptive_transitions:
        return base
    n = len(control)
    values = np.clip(np.asarray(control, dtype=float), -1, 1)
    width = max(1, round(config.transition_context_s * fps))
    edge = np.arange(1, n)
    prefix = np.r_[0, np.cumsum(values)]
    lo, hi = np.maximum(0, edge - width), np.minimum(n, edge + width)
    left = (prefix[edge] - prefix[lo]) / (edge - lo)
    right = (prefix[hi] - prefix[edge]) / (hi - edge)
    h_to_a = np.sqrt(np.maximum(left, 0) * np.maximum(-right, 0))
    a_to_h = np.sqrt(np.maximum(-left, 0) * np.maximum(right, 0))
    costs = np.broadcast_to(base, (max(0, n - 1), 3, 3)).copy()
    costs[:, 0, 1] *= 1 - config.transition_discount * h_to_a
    costs[:, 1, 0] *= 1 - config.transition_discount * a_to_h
    return costs


def infer(
    features,
    config,
    ranges,
    fps=25,
    published_pregame=None,
    learned_model=None,
    context_ranges=None,
):
    config.validate()
    positive_fps(fps)
    n = len(features["control"])
    ranges = validate_ranges(ranges, n)
    context = validate_ranges(context_ranges if context_ranges is not None else ranges, n)
    if any(not any(c <= a and b <= d for c, d in context) for a, b in ranges):
        raise ValueError("Each decoded range must be contained in one context range")
    if learned_model is None:
        scores, diagnostics = evidence(features, config, context, fps)
    else:
        if config.reliability_gated_tags or config.normalise_control:
            raise ValueError("Control gating options apply only to heuristic evidence")
        # Feature context is the full half even when decoding a held-out block. It contains
        # observations only, never held-out labels; the train purge exceeds the context radius.
        from possession.learning import learned_evidence

        scores, diagnostics = learned_evidence(features, learned_model, context, fps)
    labels = np.full(len(scores), 3, dtype=np.int8)
    # Published FAQ boundaries are an explicit input, never discovered from ground truth.
    if config.use_published_pregame and published_pregame is not None:
        a, b = published_pregame
        validate_ranges([(a, b)], n)
        scores[a:b] = (-np.inf, -np.inf, 0)
    for start, stop in ranges:
        control_feature = "motion_control" if config.motion_control else "control"
        cost = transition_costs(features[control_feature][start:stop], config, fps)
        labels[start:stop] = viterbi(scores[start:stop], cost, fps)
    return labels, diagnostics


def ablations(config):
    variants = {
        "full": config,
        "without_tags": replace(config, use_tags=False),
        "without_dead_model": replace(config, use_dead_model=False),
        "without_temporal": replace(config, team_switch_cost=0, dead_switch_cost=0),
    }
    if config.adaptive_transitions:
        variants["without_adaptive"] = replace(config, adaptive_transitions=False)
    if config.motion_control:
        variants["without_motion_control"] = replace(config, motion_control=False)
    if config.reliability_gated_tags:
        variants["without_tag_gate"] = replace(config, reliability_gated_tags=False)
    if config.normalise_control:
        variants["without_normalisation"] = replace(config, normalise_control=False)
    return variants


def match_pregame():
    # The legacy games.py kickoff_2nd notation is not the verified FAQ restart boundary.
    # starter/load_smart_tagging.py explicitly publishes the verified 113742 anchor.
    from possession.data import blocks
    from starter.load_smart_tagging import KICKOFF_VIDEO_FRAMES

    return blocks()[1][0], KICKOFF_VIDEO_FRAMES["2nd Half"]


FPS = GAMES[GAME]["fps"]
