import numpy as np
import pytest

from possession.data import blocks, graded_mask, read_tracking
from possession.features import build_features, missing_age, smooth, tag_features
from possession.model import match_pregame


def test_published_video_ranges_and_second_half_restart():
    assert blocks() == [(11185, 90652), (110808, 190198)]
    mask = graded_mask(199319)
    assert mask.sum() == 158857
    assert not mask[90652] and mask[110808]
    assert match_pregame() == (110808, 113742)


def test_tracking_one_based_ids_and_nan(tmp_path):
    path = tmp_path / "tracking.txt"
    path.write_text("1:0.1,0.2,3;NaN,NaN,NaN:0.2,0.3,4\n2:0.2,0.2,3;0.5,0.5,1:NaN,NaN,NaN\n")
    players, ball = read_tracking(path, 2, 2)
    np.testing.assert_allclose(ball[0], [0.2, 0.3, 4])
    assert np.isnan(ball[1]).all()
    assert np.isnan(players[0, 1]).all()


@pytest.mark.parametrize(
    "text",
    [
        "0:0,0,0:0,0,0\n",
        "2:0,0,0:0,0,0\n",
        "1:0,0:0,0,0\n",
        "1:0,0,0:0,0\n",
        "1:0,0,0:0,0,0\n1:0,0,0:0,0,0\n",
        "",
    ],
)
def test_tracking_rejects_malformed_and_truncated_files(tmp_path, text):
    path = tmp_path / "bad.txt"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_tracking(path, 2, 1)


def test_distance_uses_metres_not_normalised_square():
    # Home is closer in normalised coordinates; away is closer in actual metres.
    players = np.array([[[0.53, 0.5, 0], [0.5, 0.54, 0]]], dtype=float)
    ball = np.array([[0.5, 0.5, 0]])
    f = build_features(players, ball, np.array([0, 1]), [], (105.3, 68), 25, [(0, 1)])
    assert f["nearest_away"][0] < f["nearest_home"][0]
    assert f["control"][0] < 0


def test_unobserved_ball_is_not_a_zero_coordinate():
    players = np.array([[[0, 0, 0], [0.9, 0.9, 0]]], dtype=float)
    f = build_features(
        players, np.full((1, 3), np.nan), np.array([0, 1]), [], (105.3, 68), 25, [(0, 1)]
    )
    assert f["control"][0] == 0
    assert f["contact"][0] == 0
    assert not f["ball_reliable"][0]


def test_smoothing_cannot_carry_state_across_halftime():
    values = np.r_[np.ones(10), np.zeros(10), -np.ones(10)]
    result = smooth(values, 5, 25, [(0, 10), (20, 30)])
    np.testing.assert_array_equal(result, values)


def test_gap_age_respects_half_boundaries_and_no_observation():
    obs = np.array([True, False, False, True, False, False])
    age, gap = missing_age(obs, 1, [(0, 4), (4, 6)])
    np.testing.assert_array_equal(age, [0, 1, 1, 0, 2, 2])
    np.testing.assert_array_equal(gap, [0, 2, 2, 0, 2, 2])


def event(code, team):
    return {"code": code, "start_frame": 0, "end_frame": 20, "tags": {"Team": team}}


def test_defensive_tags_support_opponent_and_do_not_double_count():
    tags = [event("DEFENDING IN MIDDLE THIRD", "Barcelona"), event("PROGRESSION", "Atletic Club")]
    f = tag_features(tags * 10, 20, 25, [(0, 20)])
    np.testing.assert_array_equal(f["tag_home"], 0)
    np.testing.assert_array_equal(f["tag_away"], 1)


def test_unknown_team_set_piece_does_not_invent_owner_or_dead_label():
    f = tag_features([event("SET PIECES", "N/A")], 20, 25, [(0, 20)])
    assert f["tag_set_piece"].min() == 1
    assert f["tag_restart_home"].max() == f["tag_restart_away"].max() == 0
    assert f["tag_live"].max() == 0


def test_outlier_jump_removes_both_endpoints():
    players = np.zeros((3, 2, 3))
    ball = np.array([[0.1, 0.1, 0], [0.1, 0.1, 0], [0.9, 0.9, 0]])
    f = build_features(players, ball, np.array([0, 1]), [], (105.3, 68), 25, [(0, 3)])
    np.testing.assert_array_equal(f["ball_reliable"], [True, False, False])
