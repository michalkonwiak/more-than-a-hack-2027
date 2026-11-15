"""Reproduce only the organisers' fixed smooth50 reference, without reading any labels."""

import argparse
import importlib.metadata
import os
import platform
import tempfile
from pathlib import Path

from evaluation import possession_metrics as pm
from games import GAME, require, source_frames
from possession.data import cache_lock, sha256, write_json
from starter import baseline

ROOT = Path(__file__).resolve().parents[1]
WINDOW = 50  # The organisers' designated reference; never selected using public scores.


def reproduce(data_dir, out, verify=False):
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta, raw = map(Path, require(["atd_meta", "atd_raw"], GAME, str(data_dir)))
    inputs = {"atd_meta": sha256(meta), "atd_raw": sha256(raw)}
    reference = ROOT / "evaluation/results/baseline_smooth50.csv"
    with cache_lock(out / ".reference.lock"), tempfile.TemporaryDirectory(dir=out) as temporary:
        path = Path(temporary) / "submission.csv"

        def generate(target):
            codes = baseline.build(data_dir=str(data_dir))
            baseline.write_submission(str(target), baseline.smooth(codes, WINDOW))
            reason = pm.validate_submission(pm.read_label_csv(target), source_frames("tactical"))
            if reason:
                raise ValueError(reason)
            return sha256(target)

        fingerprint = generate(path)
        if fingerprint != sha256(reference):
            raise ValueError(
                "Generated CSV differs from the published reference; inspect inputs/code"
            )
        if verify and generate(Path(temporary) / "repeat.csv") != fingerprint:
            raise ValueError("Independent second inference produced different CSV bytes")
        if inputs != {"atd_meta": sha256(meta), "atd_raw": sha256(raw)}:
            raise ValueError("Input files changed during reproduction")
        target = out / "submission.csv"
        if target.exists() and sha256(target) != fingerprint:
            raise ValueError("Refusing to replace a different existing submission; choose --out")
        os.replace(path, target)
        report = {
            "schema": "official-reference-reproduction-v1",
            "method": "official baseline_smooth50",
            "window_half_width_frames": WINDOW,
            "selection_basis": "Organisers' designated reference, not public-window selection",
            "submission_sha256": fingerprint,
            "inputs_sha256": inputs,
            "code_sha256": {
                name: sha256(ROOT / name)
                for name in (
                    "starter/baseline.py",
                    "games.py",
                    "possession/reference.py",
                    "evaluation/possession_metrics.py",
                    "requirements.lock",
                )
            },
            "environment": {
                "python": platform.python_version(),
                **{name: importlib.metadata.version(name) for name in ("numpy", "pandas")},
            },
            "matches_published_reference": True,
            "repeated_full_inference": verify,
            "ground_truth_read": False,
            "historical_candidate_configuration_used": False,
            "quality_claim": "Reference only; no competitive improvement established",
        }
        write_json(out / "manifest.json", report)
    print(f"Verified official reference: {target}; SHA256 {fingerprint}", flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=os.environ.get("MTAH_DATA_DIR")
        or ROOT.parent / "More Than A Hack 2027 - Initial Challenge - Dataset",
    )
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/reference_baseline")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    reproduce(args.data_dir, args.out, args.verify)


if __name__ == "__main__":
    main()
