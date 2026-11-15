"""Small regularised models, with purged temporal validation and explicit training provenance.

No absolute frame, match clock, player identity or target-derived feature enters either model.
The team model is antisymmetric: swapping home and away flips its logit exactly.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

from evaluation import possession_metrics as pm
from possession.contracts import label_array
from possession.data import sha256
from possession.features import smooth

SCHEMA = "calibrated-v2"
LIVE_FEATURES = [
    f"{name}_{seconds}s"
    for seconds in (0.6, 2.4)
    for name in ("moving_fraction", "player_speed", "ball_moving", "contact", "ball_reliable")
] + ["tag_live", "tag_set_piece"]
TEAM_FEATURES = ["control_0.6s", "control_2.4s", "tag_team"]


def feature_signature():
    directory = Path(__file__).parent
    recipe = {
        name: sha256(directory / name)
        for name in ("learning.py", "features.py", "data.py", "contracts.py")
    }
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def validate_model(model):
    if isinstance(model, dict) and model.get("training_source") == (
        "public 5-minute ground truth ONLY"
    ):
        from possession.policy import refuse_retired_workflow

        refuse_retired_workflow()
    if not isinstance(model, dict) or model.get("schema") != SCHEMA:
        raise ValueError("Unsupported learned model schema; retrain using the current code")
    if model.get("live_features") != LIVE_FEATURES or model.get("team_features") != TEAM_FEATURES:
        raise ValueError("Learned model feature schema differs from this implementation")
    if model.get("feature_signature") != feature_signature():
        raise ValueError("Learned model was trained with different feature code; retrain it")
    for name, count, symmetric in (
        ("live", len(LIVE_FEATURES), False),
        ("team", len(TEAM_FEATURES), True),
    ):
        part = model.get(name)
        if not isinstance(part, dict) or part.get("symmetric") is not symmetric:
            raise ValueError(f"Invalid {name} logistic model type")
        for key, expected in (
            ("centre", count),
            ("scale", count),
            ("weights", count + (not symmetric)),
        ):
            try:
                vector = np.asarray(part.get(key), dtype=float)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid {name}.{key}") from error
            if vector.shape != (expected,) or not np.isfinite(vector).all():
                raise ValueError(f"Invalid {name}.{key} dimensions or non-finite values")
            if key == "scale" and (vector <= 0).any():
                raise ValueError(f"Invalid {name}.scale: values must be positive")
        if symmetric and np.any(np.asarray(part["centre"]) != 0):
            raise ValueError("A symmetric team model cannot have a non-zero centre")
    return model


def design(features, ranges, fps=25):
    names, columns = [], []
    for seconds in (0.6, 2.4):
        for name in ("moving_fraction", "player_speed", "ball_moving", "contact", "ball_reliable"):
            names.append(f"{name}_{seconds}s")
            columns.append(smooth(features[name], seconds, fps, ranges))
    for name in ("tag_live", "tag_set_piece"):
        names.append(name)
        columns.append(features[name])
    live = np.column_stack(columns)
    team_names = ["control_0.6s", "control_2.4s", "tag_team"]
    team = np.column_stack(
        [
            smooth(features["control"], 0.6, fps, ranges),
            smooth(features["control"], 2.4, fps, ranges),
            features["tag_home"] - features["tag_away"],
        ]
    )
    if not np.isfinite(live).all() or not np.isfinite(team).all():
        raise ValueError("Non-finite design matrix")
    return live, team, names, team_names


def fit_logistic(x, y, ridge=0.03, symmetric=False):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if (
        x.ndim != 2
        or y.shape != (len(x),)
        or not np.isfinite(x).all()
        or not np.isin(y, [0, 1]).all()
        or not np.isfinite(ridge)
        or ridge < 0
    ):
        raise ValueError("Expected finite X, binary targets and non-negative finite regularisation")
    if len(np.unique(y)) != 2:
        raise ValueError("Training requires both classes; more labelled periods are needed")
    if symmetric:
        x, y = np.r_[x, -x], np.r_[y, 1 - y]
    centre = np.zeros(x.shape[1]) if symmetric else x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 0.1)
    standard = (x - centre) / scale
    if not symmetric:
        standard = np.column_stack([standard, np.ones(len(x))])
    penalty = np.full(standard.shape[1], ridge)
    if not symmetric:
        penalty[-1] = 0  # Do not force the empirical live/dead prior to 50/50.

    def objective(w):
        z = standard @ w
        loss = np.mean(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(penalty * w**2)
        grad = standard.T @ (expit(z) - y) / len(y) + penalty * w
        return loss, grad

    result = minimize(
        objective,
        np.zeros(standard.shape[1]),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": 500, "ftol": 1e-12, "gtol": 1e-8},
    )
    if not result.success:
        raise RuntimeError(f"Logistic optimisation failed: {result.message}")
    return {
        "centre": centre.tolist(),
        "scale": scale.tolist(),
        "weights": result.x.tolist(),
        "symmetric": symmetric,
        "ridge": ridge,
        "optimisation_iterations": int(result.nit),
    }


def probability(x, model):
    standard = (x - np.asarray(model["centre"])) / np.asarray(model["scale"])
    if not model["symmetric"]:
        standard = np.column_stack([standard, np.ones(len(x))])
    return np.clip(expit(standard @ np.asarray(model["weights"])), 1e-6, 1 - 1e-6)


def fit_models(live, team, truth, train_mask, ridge=0.03):
    truth = label_array(truth, len(live))
    if train_mask.shape != truth.shape or train_mask.dtype != bool:
        raise ValueError("Training mask must be a boolean vector matching the timeline")
    if (truth[train_mask] == pm.X).any():
        raise ValueError("Training mask includes unlabelled frames")
    # Sample at 5 Hz to bound work. Independence is handled by temporal splitting, not this stride.
    indices = np.flatnonzero(train_mask)[::5]
    team_indices = indices[truth[indices] != pm.D]
    return {
        "schema": SCHEMA,
        "live_features": LIVE_FEATURES,
        "team_features": TEAM_FEATURES,
        "feature_signature": feature_signature(),
        "live": fit_logistic(live[indices], truth[indices] != pm.D, ridge),
        "team": fit_logistic(
            team[team_indices], truth[team_indices] == pm.H, ridge, symmetric=True
        ),
    }


def learned_evidence(features, model, ranges, fps=25):
    validate_model(model)
    live, team, live_names, team_names = design(features, ranges, fps)
    if model.get("live_features", live_names) != live_names:
        raise ValueError("Live feature schema differs from trained model")
    if model.get("team_features", team_names) != team_names:
        raise ValueError("Team feature schema differs from trained model")
    p_live, p_home = probability(live, model["live"]), probability(team, model["team"])
    scores = np.column_stack(
        [
            np.log(p_live) + np.log(2 * p_home),
            np.log(p_live) + np.log(2 * (1 - p_home)),
            np.log1p(-p_live),
        ]
    )
    return scores, {
        "live_evidence": p_live,
        "home_evidence": p_home,
        "observation_confidence": softmax(scores, axis=1).max(axis=1),
    }


def temporal_folds(truth, n_folds=5, purge_frames=250):
    """Contiguous validation blocks; remove neighbouring training frames within ±10 seconds."""
    if type(n_folds) is not int or n_folds < 2 or type(purge_frames) is not int or purge_frames < 0:
        raise ValueError("Need at least two folds and a non-negative integer purge")
    valid = truth != pm.X
    runs = pm.valid_runs(truth)
    if len(runs) != 1:
        raise ValueError("Initial public cross-validation expects one continuous labelled window")
    start, stop = runs[0]
    if stop - start < n_folds:
        raise ValueError("Not enough labelled frames for the requested number of folds")
    edges = np.linspace(start, stop, n_folds + 1, dtype=int)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        train = valid.copy()
        train[max(0, a - purge_frames) : min(len(truth), b + purge_frames)] = False
        if not train.any():
            raise ValueError("Purge removed all training frames")
        yield train, (int(a), int(b))


def train_and_validate(features, truth_path, out, config, data_meta):
    """The old protocol trained on this match; retained as a fail-fast compatibility stub."""
    from possession.policy import refuse_retired_workflow

    refuse_retired_workflow()


def load_model(path):
    return validate_model(json.loads(Path(path).read_text()))
