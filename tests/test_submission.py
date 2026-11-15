import numpy as np
import pytest

from evaluation import possession_metrics as pm
from possession.data import sha256
from possession.reporting import write_submission


def test_export_passes_official_validator_and_is_byte_reproducible(tmp_path):
    labels = np.array([3, 0, 0, 2, 1], dtype=np.int8)
    first = write_submission(tmp_path / "one.csv", labels, expected_rows=5)
    second = write_submission(tmp_path / "two.csv", labels, expected_rows=5)
    assert sha256(first) == sha256(second)
    assert pm.validate_submission(pm.read_label_csv(first), 5) is None
    assert first.read_bytes() == b"frame,label\n0,X\n1,H\n2,H\n3,D\n4,A\n"


def test_invalid_label_cannot_destroy_existing_submission(tmp_path):
    path = tmp_path / "submission.csv"
    path.write_text("keep me")
    with pytest.raises(ValueError):
        write_submission(path, np.array([9]))
    assert path.read_text() == "keep me"


def test_run_directory_contains_only_submission_csv_and_stats_are_per_half(tmp_path, monkeypatch):
    from possession import reporting
    from possession.model import Config

    monkeypatch.setattr(reporting, "blocks", lambda: [(0, 2), (2, 4)])
    labels = np.array([0, 0, 1, 1], dtype=np.int8)
    manifest = reporting.save_run(
        tmp_path, labels, {}, Config(), {}, {"ball_observed": np.ones(4, dtype=bool)}
    )
    assert [p.name for p in tmp_path.glob("*.csv")] == ["submission.csv"]
    assert (tmp_path / "details/segments.csv").exists()
    assert manifest["halves"][0]["possessions"]["home_share"] == 1
    assert manifest["halves"][1]["possessions"]["home_share"] == 0
