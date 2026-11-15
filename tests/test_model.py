import itertools
from dataclasses import replace

import numpy as np
import pytest

from possession.features import build_features
from possession.model import Config, infer, viterbi


@pytest.mark.parametrize("seed", range(5))
def test_viterbi_matches_exhaustive_search(seed):
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=(6, 3))
    costs = np.array([[0, 0.7, 1.2], [0.7, 0, 1.2], [1.2, 1.2, 0]])

    def objective(path):
        return sum(scores[t, state] / 2 for t, state in enumerate(path)) - sum(
            costs[a, b] for a, b in zip(path[:-1], path[1:], strict=True)
        )

    actual = viterbi(scores, costs, fps=2)
    expected = max(objective(p) for p in itertools.product(range(3), repeat=6))
    assert objective(actual) == pytest.approx(expected)


def test_one_frame_flicker_is_suppressed_but_sustained_takeover_survives():
    scores = np.full((200, 3), -5.0)
    scores[:100, 0] = 0
    scores[100:, 1] = 0
    scores[40] = [-5, 0, -5]
    cost = np.ones((3, 3)) - np.eye(3)
    path = viterbi(scores, cost)
    np.testing.assert_array_equal(path[:100], 0)
    np.testing.assert_array_equal(path[100:], 1)


def test_real_stoppage_splits_same_team_possessions():
    scores = np.full((200, 3), -5.0)
    scores[:, 0] = 0
    scores[75:125] = [-5, -5, 0]
    cost = np.ones((3, 3)) - np.eye(3)
    labels = viterbi(scores, cost)
    np.testing.assert_array_equal(labels[75:125], 2)
    assert labels[74] == labels[125] == 0


def test_no_transition_cost_is_framewise_argmax():
    scores = np.random.default_rng(13).normal(size=(100, 3))
    np.testing.assert_array_equal(viterbi(scores, np.zeros((3, 3))), scores.argmax(axis=1))


def test_decoder_rejects_non_finite_input():
    with pytest.raises(ValueError):
        viterbi(np.array([[np.nan, 0, 1]]), np.zeros((3, 3)))


@pytest.mark.parametrize(
    "kwargs", [{"team_switch_cost": -1}, {"tag_weight": float("nan")}, {"use_tags": "false"}]
)
def test_invalid_configuration_fails_early(kwargs):
    with pytest.raises(ValueError):
        replace(Config(), **kwargs).validate()


def test_prediction_has_no_ground_truth_dependency_and_obeys_masks(monkeypatch):
    from evaluation import possession_metrics as pm

    def forbidden(*args, **kwargs):
        raise AssertionError("Prediction must not load labels")

    monkeypatch.setattr(pm, "read_label_csv", forbidden)
    n = 200
    players = np.tile(np.array([[[0.5, 0.5, 3], [0.7, 0.7, 3]]]), (n, 1, 1))
    ball = np.tile([[0.5, 0.5, 3]], (n, 1))
    ranges = [(10, 90), (110, 190)]
    features = build_features(players, ball, np.array([0, 1]), [], (105.3, 68), 25, ranges)
    features["homography_error"] = np.zeros(n)
    features["homography_valid"] = np.ones(n, dtype=bool)
    labels, _ = infer(features, Config(), ranges, published_pregame=(110, 140))
    assert np.all(labels[:10] == 3) and np.all(labels[90:110] == 3)
    assert np.all(labels[110:140] == 2) and np.all(labels[190:] == 3)
