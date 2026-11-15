import numpy as np
import pytest

from possession import external_data as ed
from possession import external_model as em
from possession.data import sha256, write_json


def fixture_files(tmp_path, statuses=None):
    # Guest listed first intentionally: home is determined by Role, not XML order.
    meta = tmp_path / "meta.xml"
    meta.write_text("""<Root><General MatchId="DFL-MAT-TEST" Season="2022/2023"/>
    <Environment PitchX="100" PitchY="60"/>
    <Team TeamId="away" Role="guest"><Players><Player PersonId="a"/></Players></Team>
    <Team TeamId="home" Role="home"><Players><Player PersonId="h"/></Players></Team></Root>""")
    sections = []
    for period in ("firstHalf", "secondHalf"):
        for side, person, x in (("away", "a", 10), ("home", "h", -10), ("BALL", "ball", -9)):
            rows = []
            for i in range(4):
                state, owner = (statuses or [("1", "1"), ("1", "2"), ("0", "1"), ("1", "0")])[i]
                rows.append(
                    f'<Frame N="{100 + i}" X="{x}" Y="0" S="36" '
                    f'BallStatus="{state}" BallPossession="{owner}"/>'
                )
            sections.append(
                f'<FrameSet MatchId="DFL-MAT-TEST" GameSection="{period}" '
                f'TeamId="{side}" PersonId="{person}">{"".join(rows)}</FrameSet>'
            )
    tracking = tmp_path / "tracking.xml"
    tracking.write_text("<Root>" + "".join(sections) + "</Root>")
    return meta, tracking


def test_ingestion_team_mapping_units_periods_and_unknown_labels(tmp_path):
    meta, tracking = fixture_files(tmp_path)
    players, ball, teams, pitch, ranges, truth = ed.load_arrays(meta, tracking, "TEST")
    assert pitch == (100, 60) and ranges == [(0, 4), (4, 8)]
    assert teams.tolist() == [1, 0]
    np.testing.assert_array_equal(players[:, :, 2], 10)  # 36 km/h = 10 m/s.
    np.testing.assert_array_equal(truth, [0, 1, 2, 3, 0, 1, 2, 3])
    assert ball.shape == (8, 2)
    with pytest.raises(ValueError, match="identity"):
        ed.load_arrays(meta, tracking, "WRONG")


def test_changing_all_label_attributes_cannot_change_features(tmp_path):
    meta, tracking = fixture_files(tmp_path)
    p, b, t, pitch, ranges, truth = ed.load_arrays(meta, tracking, "TEST")
    before = em.observation_features(p, b, t, pitch, ranges)
    meta, tracking = fixture_files(tmp_path, [("0", "2")] * 4)
    p2, b2, t2, pitch2, ranges2, truth2 = ed.load_arrays(meta, tracking, "TEST")
    after = em.observation_features(p2, b2, t2, pitch2, ranges2)
    assert not np.array_equal(truth, truth2)
    np.testing.assert_array_equal(before[0], after[0])
    np.testing.assert_array_equal(before[1], after[1])


def test_features_are_team_symmetric_and_period_isolated(tmp_path):
    meta, tracking = fixture_files(tmp_path)
    p, b, t, pitch, ranges, _ = ed.load_arrays(meta, tracking, "TEST")
    live, team, names = em.observation_features(p, b, t, pitch, ranges)
    live2, team2, _ = em.observation_features(p, b, 1 - t, pitch, ranges)
    np.testing.assert_allclose(live, live2)
    np.testing.assert_allclose(team, -team2)
    b[4:] *= 100
    p[4:] *= 10
    changed = em.observation_features(p, b, t, pitch, ranges)
    np.testing.assert_array_equal(live[:4], changed[0][:4])
    assert len(names["live"]) == live.shape[1]
    assert len(names["team"]) == team.shape[1]
    missing = em.observation_features(p, np.full_like(b, np.nan), t, pitch, ranges)
    assert np.isfinite(missing[0]).all() and np.isfinite(missing[1]).all()


def test_duplicate_frames_rejected_and_ball_status_precedes_ownership(tmp_path):
    assert ed.label("0", "2") == 2
    assert ed.label("1", "0") == 3
    assert ed.label(None, "1") == 3
    meta, tracking = fixture_files(tmp_path)
    text = tracking.read_text()
    text = text.replace('N="101"', 'N="100"')
    tracking.write_text(text)
    with pytest.raises(ValueError, match="Duplicate ball"):
        ed.load_arrays(meta, tracking, "TEST")


def test_only_pinned_external_matches_and_disjoint_splits_are_accepted():
    spec = ed.protocol()
    matches = sum(spec["splits"].values(), [])
    assert len(matches) == len(set(matches)) == 7
    with pytest.raises(ValueError, match="Unknown external"):
        ed.source_records("BAR-ATH")
    first = ed.dropout_mask(1000, [(0, 500), (500, 1000)], "J03WMX")
    second = ed.dropout_mask(1000, [(0, 500), (500, 1000)], "J03WMX")
    np.testing.assert_array_equal(first, second)
    assert 0 < first.sum() < 1000


def test_external_source_digest_verification(tmp_path):
    path = tmp_path / "file"
    path.write_bytes(b"sample")
    record = {"size": 6, "sha256": sha256(path)}
    ed.verify_source(path, record)
    path.write_bytes(b"change")
    with pytest.raises(ValueError, match="fingerprint"):
        ed.verify_source(path, record)


def test_two_halves_do_not_merge_possessions():
    result = em.evaluate(
        np.zeros(100, dtype=np.int8), np.zeros(100, dtype=np.int8), [(0, 50), (50, 100)]
    )
    assert result["predicted_possessions"] == 2
    assert result["spell_f1"] == 1


def test_holdout_refuses_changed_freeze_before_loading_any_data(tmp_path, monkeypatch):
    write_json(tmp_path / "selected.json", {"schema": "external-logistic-v1"})
    write_json(tmp_path / "freeze.json", {"selected_sha256": "wrong"})
    monkeypatch.setattr(em, "cache", lambda *a: pytest.fail("Holdout read too early"))
    with pytest.raises(ValueError, match="Frozen"):
        em.test_holdout(tmp_path, tmp_path)


def test_train_select_test_workflow_is_match_separated(tmp_path, monkeypatch):
    spec = {
        **ed.protocol(),
        "splits": {"train": ["train-game"], "development": ["dev-game"], "holdout": ["test-game"]},
        "ridge_candidates": [0.1],
        "transition_cost_candidates": [0.0],
    }
    monkeypatch.setattr(em, "protocol", lambda: spec)
    calls = []
    rng = np.random.default_rng(11)
    truth = np.repeat([0, 1, 2], 100).astype(np.int8)
    x = rng.normal(size=(300, 3))
    x[:, 0] += (truth != 2) * 2
    y = rng.normal(size=(300, 2))
    y[:, 0] += np.where(truth == 0, 2, -2)
    data = {"truth": truth, "ranges": np.array([(0, 150), (150, 300)])}
    for condition in spec["conditions"]:
        data.update(
            {f"live_{condition}": x, f"team_{condition}": y, f"reference_{condition}": truth.copy()}
        )

    def cache(match, root):
        calls.append(match)
        if match == "test-game":
            assert (tmp_path / "selected.json").exists() and (tmp_path / "freeze.json").exists()
        p = root / f"{match}.npz"
        p.write_bytes(b"synthetic cache")
        return data, sha256(p)

    monkeypatch.setattr(em, "cache", cache)
    selected = em.train(tmp_path, tmp_path)
    assert calls == ["train-game", "dev-game"]
    assert selected["competition_data_used"] is False
    assert selected["competition_release_approved"] is False
    report = em.test_holdout(tmp_path, tmp_path)
    assert calls[-1] == "test-game"
    assert report["reference_mean_spell_f1"] == 1
    with pytest.raises(ValueError, match="already evaluated"):
        em.test_holdout(tmp_path, tmp_path)
    with pytest.raises(ValueError, match="frozen"):
        em.train(tmp_path, tmp_path)
