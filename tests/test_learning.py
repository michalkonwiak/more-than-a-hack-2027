import numpy as np
import pytest

from possession.learning import fit_logistic, probability, temporal_folds


def test_symmetric_team_model_is_invariant_to_swapping_team_names():
    rng = np.random.default_rng(13)
    x = rng.normal(size=(100, 3))
    y = (x[:, 0] + 0.5 * x[:, 1]) > 0
    model = fit_logistic(x, y, symmetric=True)
    np.testing.assert_allclose(probability(x, model), 1 - probability(-x, model), atol=1e-12)


def test_fit_uses_training_statistics_only_and_is_reproducible():
    x = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y = np.array([0, 0, 1, 1])
    first, second = fit_logistic(x, y), fit_logistic(x, y)
    assert first == second
    assert first["centre"] == [0.0]
    assert probability(x, first)[0] < 0.5 < probability(x, first)[-1]


def test_constant_target_is_rejected_instead_of_reporting_false_validation():
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic(np.ones((20, 3)), np.ones(20))


def test_temporal_folds_have_no_label_leakage_and_cover_every_public_frame_once():
    truth = np.full(9000, 3)
    truth[500:8000] = np.arange(7500) % 3
    covered = np.zeros(9000)
    for train, (a, b) in temporal_folds(truth):
        assert not train[max(0, a - 250) : b + 250].any()
        assert not train[truth == 3].any()
        covered[a:b] += 1
    np.testing.assert_array_equal(covered[truth != 3], 1)
    np.testing.assert_array_equal(covered[truth == 3], 0)
