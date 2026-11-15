"""Experimental transfer of the frozen external model; never reads competition labels."""

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from games import GAME, GAMES, require, source_frames
from possession.data import blocks, cache_lock, read_tracking, sha256, write_json
from possession.external_data import PROTOCOL, protocol
from possession.external_model import (
    decode,
    implementation_hashes,
    observation_features,
    scores,
)
from possession.reporting import write_submission
from starter.baseline import team_of_each_track

ROOT = Path(__file__).resolve().parents[1]


def frozen_model(path):
    model = json.loads(path.read_text())
    freeze = json.loads(path.with_name("freeze.json").read_text())
    if freeze != {"selected_sha256": sha256(path), "protocol_sha256": sha256(PROTOCOL)}:
        raise ValueError("External model or protocol no longer matches its freeze")
    spec = protocol()
    if (
        model.get("schema") != "external-logistic-v1"
        or model.get("competition_data_used") is not False
        or model.get("training_matches") != spec["splits"]["train"]
        or model.get("selection_matches") != spec["splits"]["development"]
    ):
        raise ValueError("Expected the externally trained and selected model")
    if model["implementation_sha256"] != implementation_hashes():
        raise ValueError("External implementation changed after selection")
    if model["development_sha256"] != sha256(path.with_name("development.json")):
        raise ValueError("Development report changed after selection")
    return model


def centred_metres(players, ball, pitch):
    """ATD positions are 0..1; ATD player speeds already use m/s (official DATA.md)."""
    players = players.copy()
    players[:, :, :2] = (players[:, :, :2] - 0.5) * np.asarray(pitch)
    ball = (ball[:, :2] - 0.5) * np.asarray(pitch)
    return players, ball


def predict_once(meta, raw, selected):
    sides = team_of_each_track(meta)
    players, ball = read_tracking(raw, source_frames("tactical"), len(sides))
    pitch = GAMES[GAME]["pitch"]
    players, ball = centred_metres(players, ball, pitch)
    ranges = blocks()
    live, team, _ = observation_features(players, ball, sides, pitch, ranges)
    # External caches use float32. Apply the same feature precision at transfer time.
    evidence = scores(live.astype(np.float32), team.astype(np.float32), selected["model"])
    return decode(evidence, selected["transition_cost"], ranges)


def predict(data_dir, model_path, out, verify=False):
    selected = frozen_model(model_path)
    model_hash = sha256(model_path)
    meta, raw = map(Path, require(["atd_meta", "atd_raw"], GAME, str(data_dir)))
    inputs = {"atd_meta": sha256(meta), "atd_raw": sha256(raw)}
    out.mkdir(parents=True, exist_ok=True)
    with cache_lock(out / ".run.lock"), tempfile.TemporaryDirectory(dir=out) as temporary:
        path = Path(temporary) / "submission.csv"
        labels = predict_once(meta, raw, selected)
        write_submission(path, labels)
        digest = sha256(path)
        if verify:
            again = predict_once(meta, raw, selected)
            second = Path(temporary) / "repeated.csv"
            write_submission(second, again)
            if sha256(second) != digest:
                raise ValueError("Repeated external inference produced different CSV bytes")
        if inputs != {"atd_meta": sha256(meta), "atd_raw": sha256(raw)}:
            raise ValueError("Competition observations changed during inference")
        frozen_model(model_path)
        if sha256(model_path) != model_hash:
            raise ValueError("External model changed during inference")
        target = out / "submission.csv"
        if target.exists() and sha256(target) != digest:
            raise ValueError("Refusing to overwrite different candidate bytes; choose a new --out")
        os.replace(path, target)
        report = {
            "schema": "external-transfer-v1",
            "status": "experimental_candidate",
            "model_sha256": model_hash,
            "protocol_sha256": sha256(PROTOCOL),
            "inputs_sha256": inputs,
            "submission_sha256": digest,
            "implementation_sha256": {
                **implementation_hashes(),
                **{
                    name: sha256(ROOT / name)
                    for name in (
                        "possession/external_infer.py",
                        "possession/data.py",
                        "possession/reporting.py",
                        "possession/contracts.py",
                        "games.py",
                        "starter/baseline.py",
                        "evaluation/possession_metrics.py",
                        "requirements.lock",
                    )
                },
            },
            "ground_truth_read_during_inference": False,
            "competition_labels_used_for_training_or_selection": False,
            "historical_candidate_parameters_used": False,
            "repeated_full_inference": verify,
            "competition_release_approved": False,
            "label_counts": {str(i): int((labels == i).sum()) for i in range(4)},
            "caveat": "Frozen external model applied with unit conversion only. Target tracking "
            "noise differs from external data. No private/public target accuracy is claimed; "
            "earlier project history remains disclosed in PROVENANCE_AUDIT.md.",
        }
        write_json(out / "manifest.json", report)
    print(f"Experimental external candidate: {target}; SHA256 {digest}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", type=Path, default=ROOT / "models/external_idsse_v1/selected.json"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/candidates/external_idsse_v1")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("MTAH_DATA_DIR")
        or ROOT.parent / "More Than A Hack 2027 - Initial Challenge - Dataset",
    )
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    predict(args.data_dir, args.model, args.out, args.verify)


if __name__ == "__main__":
    main()
