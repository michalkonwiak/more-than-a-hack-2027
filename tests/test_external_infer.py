import json

import numpy as np
import pytest

from possession import external_infer as ei
from possession.data import sha256, write_json
from possession.external_model import observation_features
from possession.reporting import write_submission


def test_atd_conversion_preserves_speeds_and_missingness():
    players = np.array([[[0.0, 0.0, 3.0], [1.0, 1.0, 4.0]]], dtype=np.float32)
    ball = np.array([[0.5, np.nan, 99.0]])
    p, b = ei.centred_metres(players, ball, (100, 60))
    np.testing.assert_array_equal(p[0], [[-50, -30, 3], [50, 30, 4]])
    assert b[0, 0] == 0 and np.isnan(b[0, 1])
    np.testing.assert_array_equal(players[0, 0], [0, 0, 3])


def test_axis_reflection_does_not_change_external_features():
    rng = np.random.default_rng(25)
    p = rng.normal(size=(50, 4, 3))
    p[:, :, 2] = np.abs(p[:, :, 2])
    b = rng.normal(size=(50, 2))
    sides = np.array([0, 0, 1, 1])
    before = observation_features(p, b, sides, (105, 68), [(0, 50)])
    p[:, :, 1] *= -1
    b[:, 1] *= -1
    after = observation_features(p, b, sides, (105, 68), [(0, 50)])
    np.testing.assert_allclose(before[0], after[0])
    np.testing.assert_allclose(before[1], after[1])


def model_files(tmp_path):
    write_json(tmp_path / "development.json", {"fixture": True})
    spec = ei.protocol()
    model = {
        "schema": "external-logistic-v1",
        "competition_data_used": False,
        "training_matches": spec["splits"]["train"],
        "selection_matches": spec["splits"]["development"],
        "implementation_sha256": ei.implementation_hashes(),
        "development_sha256": sha256(tmp_path / "development.json"),
    }
    path = tmp_path / "selected.json"
    write_json(path, model)
    write_json(
        tmp_path / "freeze.json",
        {"selected_sha256": sha256(path), "protocol_sha256": sha256(ei.PROTOCOL)},
    )
    return path


def test_model_provenance_and_freeze_are_checked(tmp_path):
    path = model_files(tmp_path)
    assert ei.frozen_model(path)["competition_data_used"] is False
    model = json.loads(path.read_text())
    model["competition_data_used"] = True
    write_json(path, model)
    with pytest.raises(ValueError, match="freeze"):
        ei.frozen_model(path)
    write_json(
        path.with_name("freeze.json"),
        {"selected_sha256": sha256(path), "protocol_sha256": sha256(ei.PROTOCOL)},
    )
    with pytest.raises(ValueError, match="externally trained"):
        ei.frozen_model(path)


def test_candidate_repeats_inference_and_preserves_different_output(tmp_path, monkeypatch):
    path = model_files(tmp_path)
    files = [tmp_path / "meta", tmp_path / "raw"]
    for file in files:
        file.write_text("synthetic observations")
    monkeypatch.setattr(ei, "require", lambda *a: files)
    calls = []

    def predict_once(*args):
        calls.append(True)
        return np.array([0, 1, 2, 3, 1, 0], dtype=np.int8)

    monkeypatch.setattr(ei, "predict_once", predict_once)
    monkeypatch.setattr(
        ei, "write_submission", lambda p, y: write_submission(p, y, expected_rows=6)
    )
    out = tmp_path / "out"
    report = ei.predict(tmp_path, path, out, verify=True)
    assert len(calls) == 2
    assert report["repeated_full_inference"]
    assert not report["competition_labels_used_for_training_or_selection"]
    assert not report["competition_release_approved"]
    (out / "submission.csv").write_text("preserve")
    with pytest.raises(ValueError, match="overwrite"):
        ei.predict(tmp_path, path, out)
    assert (out / "submission.csv").read_text() == "preserve"
