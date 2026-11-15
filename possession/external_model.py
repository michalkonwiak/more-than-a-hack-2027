"""Match-separated IDSSE pilot; no competition inference or public-label scoring."""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import uniform_filter1d

from evaluation import possession_metrics as pm
from possession.data import sha256, write_json
from possession.external_data import DATA, PROTOCOL, prepare_match, protocol
from possession.learning import fit_logistic, probability
from possession.model import viterbi
from starter.baseline import smooth as reference_smooth

RUN = Path(__file__).resolve().parents[1] / "artifacts/external_runs/idsse_v1"
LIVE_NAMES = [
    "closest_log_distance",
    "farthest_team_log_distance",
    "absolute_x",
    "absolute_y",
    "log_ball_speed",
    "mean_player_speed",
    "player_speed_sd",
    "player_coverage",
    "ball_observed",
]
TEAM_NAMES = ["log_distance_advantage", "relative_distance_advantage", "player_count_advantage"]


def observation_features(players, ball, teams, pitch, ranges):
    """Coordinates/speeds only. Neither possession nor live/dead labels are arguments."""
    n = len(ball)
    observed = np.isfinite(ball).all(axis=1)
    valid = np.isfinite(players[:, :, :2]).all(axis=2)
    distance = np.linalg.norm(players[:, :, :2] - ball[:, None, :], axis=2)
    distance[~valid | ~observed[:, None]] = np.inf
    near = np.column_stack([np.min(distance[:, teams == k], axis=1) for k in (0, 1)])
    both = np.isfinite(near).all(axis=1)
    near[~both] = 0
    speed = np.zeros(n)
    for a, b in ranges:
        delta = np.linalg.norm(np.diff(ball[a:b], axis=0), axis=1) * 25
        speed[a + 1 : b] = np.nan_to_num(delta, nan=0, posinf=0)
    velocities = players[:, :, 2]
    good_speed = valid & np.isfinite(velocities) & (velocities >= 0)
    count = good_speed.sum(axis=1)
    total = np.where(good_speed, velocities, 0).sum(axis=1)
    mean = total / np.maximum(count, 1)
    variance = np.where(good_speed, (velocities - mean[:, None]) ** 2, 0).sum(axis=1)
    sd = np.sqrt(variance / np.maximum(count, 1))
    xy = np.nan_to_num(np.abs(ball) / (np.array(pitch) / 2), nan=0)
    live = np.column_stack(
        [
            np.log1p(near.min(axis=1)),
            np.log1p(near.max(axis=1)),
            xy,
            np.log1p(speed),
            mean,
            sd,
            valid.sum(axis=1) / 22,
            observed,
        ]
    )
    team = np.column_stack(
        [
            np.log1p(near[:, 1]) - np.log1p(near[:, 0]),
            (near[:, 1] - near[:, 0]) / (near.sum(axis=1) + 1),
            (valid[:, teams == 0].sum(axis=1) - valid[:, teams == 1].sum(axis=1)) / 22,
        ]
    )
    # Missing team geometry supplies no evidence for either side.
    team[:, :2] *= both[:, None]
    names = {}
    matrices = []
    for name, base, columns in (("live", live, LIVE_NAMES), ("team", team, TEAM_NAMES)):
        values, labels = [base], [f"{c}_instant" for c in columns]
        for seconds in protocol()["feature_windows_seconds"]:
            averaged = np.zeros_like(base)
            for a, b in ranges:
                averaged[a:b] = uniform_filter1d(
                    base[a:b], round(seconds * 25), axis=0, mode="nearest"
                )
            values.append(averaged)
            labels.extend(f"{c}_mean{seconds}s" for c in columns)
        matrix = np.column_stack(values)
        if not np.isfinite(matrix).all():
            raise ValueError("Non-finite external features")
        matrices.append(matrix)
        names[name] = labels
    return *matrices, names


def reference_labels(players, ball, teams, ranges):
    distance = np.linalg.norm(players[:, :, :2] - ball[:, None, :], axis=2)
    distance[~np.isfinite(distance)] = np.inf
    codes = teams[distance.argmin(axis=1)].copy()
    codes[~np.isfinite(distance.min(axis=1))] = -1
    out = np.full(len(ball), 3, dtype=np.int8)
    for a, b in ranges:
        labels = reference_smooth(codes[a:b], 50)
        out[a:b] = np.where(labels == "H", 0, np.where(labels == "A", 1, 2))
    return out


def scores(live, team, model):
    p_live = probability(live, model["live"])
    p_home = probability(team, model["team"])
    return np.log(np.column_stack([p_live * p_home, p_live * (1 - p_home), 1 - p_live]))


def decode(evidence, cost, ranges):
    out = np.full(len(evidence), 3, dtype=np.int8)
    matrix = np.full((3, 3), cost) - np.eye(3) * cost
    for a, b in ranges:
        out[a:b] = viterbi(evidence[a:b], matrix, fps=25)
    return out


def evaluate(truth, prediction, ranges):
    # Separate halves explicitly: adjacent H spells across halftime cannot merge.
    truth = np.concatenate([np.r_[truth[a:b], pm.X] for a, b in ranges])
    prediction = np.concatenate([np.r_[prediction[a:b], pm.X] for a, b in ranges])
    metrics = pm.evaluate(truth, prediction, bootstrap_n=0)
    return {
        "spell_f1": metrics["spell_f1_mean"],
        "macro_f1": metrics["macro_f1"],
        "known_frames": int((truth != pm.X).sum()),
        "unknown_frames": int((truth == pm.X).sum()) - len(ranges),
        "predicted_possessions": metrics["n_pred_spells"],
    }


def cache(match, root):
    path = prepare_match(match, root)
    with np.load(path, allow_pickle=False) as data:
        result = {k: data[k] for k in data.files}
    return result, sha256(path)


def implementation_hashes():
    directory = Path(__file__).parent
    return {
        name: sha256(directory / name)
        for name in (
            "external_data.py",
            "external_model.py",
            "learning.py",
            "model.py",
        )
    }


def train(root=DATA, out=RUN):
    spec = protocol()
    if (out / "selected.json").exists() or (out / "holdout.json").exists():
        raise ValueError("This experiment is frozen; do not retune after selection/holdout")
    inputs, matrices = {}, {"live": [], "team": [], "truth": []}
    stride = spec["training_stride"]
    for match in spec["splits"]["train"]:
        data, inputs[match] = cache(match, root)
        for condition in spec["conditions"]:
            idx = np.arange(0, len(data["truth"]), stride)
            idx = idx[data["truth"][idx] != pm.X]
            matrices["live"].append(data[f"live_{condition}"][idx])
            matrices["team"].append(data[f"team_{condition}"][idx])
            matrices["truth"].append(data["truth"][idx])
    live, team, truth = (np.concatenate(matrices[k]) for k in ("live", "team", "truth"))
    del matrices
    development = {}
    for match in spec["splits"]["development"]:
        development[match], inputs[match] = cache(match, root)
    out.mkdir(parents=True, exist_ok=True)
    rows, models = [], {}
    for ridge in spec["ridge_candidates"]:
        active = truth != pm.D
        model = {
            "live": fit_logistic(live, truth != pm.D, ridge=ridge),
            "team": fit_logistic(team[active], truth[active] == pm.H, ridge=ridge, symmetric=True),
        }
        models[ridge] = model
        for cost in spec["transition_cost_candidates"]:
            results = []
            for match, data in development.items():
                for condition in spec["conditions"]:
                    evidence = scores(data[f"live_{condition}"], data[f"team_{condition}"], model)
                    labels = decode(evidence, cost, data["ranges"])
                    results.append(
                        {
                            "match": match,
                            "condition": condition,
                            **evaluate(data["truth"], labels, data["ranges"]),
                        }
                    )
            mean = float(np.mean([r["spell_f1"] for r in results]))
            rows.append(
                {
                    "ridge": ridge,
                    "transition_cost": cost,
                    "selection_score": mean,
                    "results": results,
                }
            )
            print(
                f"External development: ridge={ridge}, cost={cost}, spellF1={mean:.4f}", flush=True
            )
    best = max(rows, key=lambda r: r["selection_score"])
    references = []
    for match, data in development.items():
        for condition in spec["conditions"]:
            references.append(
                {
                    "match": match,
                    "condition": condition,
                    **evaluate(data["truth"], data[f"reference_{condition}"], data["ranges"]),
                }
            )
    write_json(out / "development.json", {"candidates": rows, "reference": references})
    selected = {
        "schema": "external-logistic-v1",
        "protocol_sha256": sha256(PROTOCOL),
        "implementation_sha256": implementation_hashes(),
        "input_cache_sha256": inputs,
        "training_matches": spec["splits"]["train"],
        "selection_matches": spec["splits"]["development"],
        "ridge": best["ridge"],
        "transition_cost": best["transition_cost"],
        "model": models[best["ridge"]],
        "development_sha256": sha256(out / "development.json"),
        "competition_data_used": False,
        "competition_release_approved": False,
        "caveat": "External pilot; provider labels and cleaner tracking differ from competition. "
        "A fixed artificial dropout is not a full simulation of tracking errors.",
    }
    write_json(out / "selected.json", selected)
    write_json(
        out / "freeze.json",
        {"selected_sha256": sha256(out / "selected.json"), "protocol_sha256": sha256(PROTOCOL)},
    )
    print("External winner frozen; holdout labels have not been read.", flush=True)
    return selected


def test_holdout(root=DATA, out=RUN):
    spec = protocol()
    selected_path = out / "selected.json"
    freeze = json.loads((out / "freeze.json").read_text())
    if freeze != {"selected_sha256": sha256(selected_path), "protocol_sha256": sha256(PROTOCOL)}:
        raise ValueError("Frozen model or protocol changed before holdout evaluation")
    model = json.loads(selected_path.read_text())
    if (out / "holdout.json").exists():
        raise ValueError("Holdout already evaluated; preserve the report and do not retune")
    if model["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("Protocol changed after model selection")
    if model["implementation_sha256"] != implementation_hashes():
        raise ValueError("Code changed after model selection")
    if model["development_sha256"] != sha256(out / "development.json"):
        raise ValueError("Development report changed after model selection")
    for match, digest in model["input_cache_sha256"].items():
        if sha256(root / f"{match}.npz") != digest:
            raise ValueError("Training/development cache changed after selection")
    fingerprint = sha256(selected_path)
    rows, inputs = [], {}
    for match in spec["splits"]["holdout"]:
        data, inputs[match] = cache(match, root)
        for condition in spec["conditions"]:
            evidence = scores(data[f"live_{condition}"], data[f"team_{condition}"], model["model"])
            prediction = decode(evidence, model["transition_cost"], data["ranges"])
            rows.append(
                {
                    "match": match,
                    "condition": condition,
                    "candidate": evaluate(data["truth"], prediction, data["ranges"]),
                    "reference": evaluate(
                        data["truth"], data[f"reference_{condition}"], data["ranges"]
                    ),
                }
            )
    if fingerprint != sha256(selected_path):
        raise ValueError("Frozen model changed during holdout evaluation")
    report = {
        "selected_sha256": fingerprint,
        "holdout_cache_sha256": inputs,
        "results": rows,
        "candidate_mean_spell_f1": float(np.mean([r["candidate"]["spell_f1"] for r in rows])),
        "reference_mean_spell_f1": float(np.mean([r["reference"]["spell_f1"] for r in rows])),
        "caveat": "Two held-out matches, both with the same home club. External provider "
        "labels are a different benchmark; this is not the private competition score.",
    }
    write_json(out / "holdout.json", report)
    print(
        f"Holdout mean spellF1: candidate={report['candidate_mean_spell_f1']:.4f}, "
        f"reference={report['reference_mean_spell_f1']:.4f}",
        flush=True,
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "test"])
    parser.add_argument("--root", type=Path, default=DATA)
    parser.add_argument("--out", type=Path, default=RUN)
    args = parser.parse_args()
    if args.command == "train":
        train(args.root, args.out)
    else:
        test_holdout(args.root, args.out)


if __name__ == "__main__":
    main()
