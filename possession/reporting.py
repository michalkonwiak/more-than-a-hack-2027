"""Evaluation is intentionally separate from inference; outputs stay under artifacts/."""

import csv
import importlib.metadata
import json
import os
import platform
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np

from evaluation import possession_metrics as pm
from games import GAME, GAMES
from possession.contracts import label_array
from possession.data import blocks, cache_lock, sha256, write_json, write_npz

ROOT = Path(__file__).resolve().parents[1]


def write_submission(path, labels, expected_rows=GAMES[GAME]["n_video_frames"]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = label_array(labels, expected_rows)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["frame", "label"])
        writer.writerows((i, pm.LABELS[int(code)]) for i, code in enumerate(labels))
    try:
        reason = pm.validate_submission(pm.read_label_csv(temporary), expected_rows)
        if reason:
            raise ValueError(reason)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def code_manifest():
    files = [ROOT / "games.py", ROOT / "requirements.lock"]
    for folder in ("possession", "starter", "evaluation"):
        files.extend(sorted((ROOT / folder).glob("*.py")))
    files.extend(sorted((ROOT / "possession").glob("*.html")))
    return {str(p.relative_to(ROOT)): sha256(p) for p in files}


def save_run(out, labels, diagnostics, config, data_meta, features, learned_model=None):
    out = Path(out)
    expected_rows = data_meta.get("n_frames", len(labels))
    label_array(labels, expected_rows)
    if any(
        np.asarray(v).shape != (expected_rows,) or not np.isfinite(v).all()
        for v in diagnostics.values()
    ):
        raise ValueError("Diagnostics must be finite vectors matching the submission timeline")
    if len(features["ball_observed"]) != expected_rows:
        raise ValueError("Observation features do not match the submission timeline")
    config.validate()
    out.mkdir(parents=True, exist_ok=True)
    with cache_lock(out / ".run.lock"):
        return _save_run(out, labels, diagnostics, config, data_meta, features, learned_model)


def _save_run(out, labels, diagnostics, config, data_meta, features, learned_model):
    csv_path = write_submission(out / "submission.csv", labels, expected_rows=len(labels))
    write_npz(out / "diagnostics.npz", diagnostics)
    summary = {
        "configuration": asdict(config),
        "data": data_meta,
        "submission_sha256": sha256(csv_path),
        "code": code_manifest(),
        "environment": {
            "python": platform.python_version(),
            **{p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")},
        },
        "label_counts": {label: int((labels == i).sum()) for i, label in enumerate(pm.LABELS)},
        "halves": [],
        "ground_truth_read_during_inference": False,
        "learned_model": learned_model,
        "confidence_is_calibrated": False,
    }
    for start, stop in blocks():
        # The official helper's home_share counts its whole array, so slice before calling.
        stats = pm.possession_stats(labels[start:stop], [(0, stop - start)])
        summary["halves"].append(
            {
                "first_frame": start,
                "stop_frame": stop,
                "possessions": stats,
                "dead_fraction": float((labels[start:stop] == pm.D).mean()),
                "ball_observed_fraction": float(features["ball_observed"][start:stop].mean()),
            }
        )
    details = out / "details"
    details.mkdir(exist_ok=True)
    with (details / "segments.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["start_frame", "stop_frame", "label", "duration_s"])
        writer.writerows(
            (a, b, pm.LABELS[k], (b - a) / 25) for a, b, k in pm.segments(labels, blocks())
        )
    summary["artifacts_sha256"] = {
        name: sha256(out / name)
        for name in ("submission.csv", "diagnostics.npz", "details/segments.csv")
    }
    # Publish the manifest last. Readers verify all referenced files, so an interrupted write
    # cannot silently pair old metadata with new predictions or diagnostic evidence.
    write_json(out / "manifest.json", summary)
    return summary


def verify_run(out):
    out = Path(out)
    report = json.loads((out / "manifest.json").read_text())
    expected = {"submission.csv", "diagnostics.npz", "details/segments.csv"}
    hashes = report.get("artifacts_sha256", {})
    if set(hashes) != expected:
        raise ValueError("Run lacks an integrity manifest; regenerate it with current code")
    if any(sha256(out / name) != hashes[name] for name in expected):
        raise ValueError("Run artifact integrity check failed; regenerate this run")
    if report["submission_sha256"] != hashes["submission.csv"]:
        raise ValueError("Conflicting submission hashes in run manifest")
    return report


def evaluate_public(predictions, out, bootstrap=1000, gt_path=None):
    """Apply the unmodified official metric, without selecting or tuning a configuration."""
    gt_path = Path(gt_path or ROOT / "evaluation/gt/BAR-ATH_possession_gt_public.csv")
    truth = pm.read_label_csv(gt_path)
    reason = pm.validate_submission(truth, 199319)
    if reason:
        raise ValueError(f"Invalid validation labels: {reason}")
    truth = pm.label_codes(truth)
    if not predictions:
        raise ValueError("At least one prediction is required")
    for labels in predictions.values():
        label_array(labels, len(truth))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    reports = {}
    for name, labels in predictions.items():
        result = pm.evaluate(truth, labels, bootstrap_n=bootstrap)
        reports[name] = result
        write_json(out / f"{name}.json", result)
        rows.append(
            {
                "name": name,
                **{
                    key: result[key]
                    for key in (
                        "spell_f1_mean",
                        "macro_f1",
                        "n_pred_spells",
                        "fragmentation",
                        "spell_bootstrap_low",
                        "spell_bootstrap_high",
                    )
                },
                "dead_f1": result["per_class"]["D"]["f1"],
            }
        )
    with (out / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    text = [
        "# Local evaluation",
        "",
        "The public window contains only 17 possessions. These are development diagnostics,",
        "not estimates of the private full-match score.",
        "No configuration is selected automatically.",
        "Bootstrap bounds are the official 5th–95th percentiles, not a 95% confidence interval.",
        "",
        "| Model | spellF1 | macroF1 | D F1 | Pred. possessions | Bootstrap 5–95% |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lo, hi = row["spell_bootstrap_low"], row["spell_bootstrap_high"]
        interval = "not requested" if lo is None else f"{lo:.4f}–{hi:.4f}"
        text.append(
            f"| {row['name']} | {row['spell_f1_mean']:.4f} | {row['macro_f1']:.4f} "
            f"| {row['dead_f1']:.4f} | {row['n_pred_spells']} | {interval} |"
        )
    (out / "comparison.md").write_text("\n".join(text) + "\n")
    return rows, reports


def validation_plan(features, out, seed=2027):
    """Small, fixed stratified video sample. No label suggestions; no automatic 'truth'."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    existing = out / "validation_plan.json"
    if existing.exists():
        return json.loads(existing.read_text())
    rng = np.random.default_rng(seed)
    clips = []
    duration, buffer = 40 * 25, 10 * 25
    for half, (start, stop) in enumerate(blocks(), 1):
        start = max(start, 18685 + buffer) if half == 1 else 113742 + buffer
        candidates = np.arange(start, stop - duration - buffer, 10 * 25)
        rng.shuffle(candidates)
        for stratum in ("restart", "occlusion", "ordinary"):
            for split in ("development", "holdout"):
                for a in candidates:
                    a, b = int(a), int(a + duration)
                    if any(
                        a < c["stop_frame"] + buffer and b > c["start_frame"] - buffer
                        for c in clips
                    ):
                        continue
                    missing = 1 - features["ball_observed"][a:b].mean()
                    restart = features["tag_set_piece"][a:b].mean()
                    eligible = {
                        "restart": restart > 0.05,
                        "occlusion": missing > 0.4 and restart == 0,
                        "ordinary": missing < 0.25 and restart == 0,
                    }[stratum]
                    if eligible:
                        clips.append(
                            {
                                "half": half,
                                "split": split,
                                "stratum": stratum,
                                "start_frame": a,
                                "stop_frame": b,
                                "context_start": a - buffer,
                                "context_stop": b + buffer,
                            }
                        )
                        break
                else:
                    raise ValueError(f"Cannot sample independent clip for {half}/{stratum}/{split}")
    plan = {
        "seed": seed,
        "fps": 25,
        "label_source": "NOT ANNOTATED",
        "clips": clips,
        "warning": "Human validation is a proxy for corrected-tracking labels.",
    }
    write_json(existing, plan)
    (out / "PROTOCOL.md").write_text(
        "# Validation protocol\n\n"
        "This fixed sample contains 12 clips of 40 seconds (8 minutes), balanced across halves, "
        "ordinary play, missing ball and restarts. Each clip has 10 seconds of context.\n\n"
        "Annotate without viewing model predictions. RULES.md allows small training/validation "
        "annotations, not manual submission correction. Mark uncertain boundaries separately. "
        "Review disagreements with a second team member where possible.\n\n"
        "Use development clips for diagnosis. Keep holdout annotations hidden until configuration "
        "freeze. Never pass annotations into prediction, create frame-specific exceptions or "
        "copy labels into the submission. Partial possession boundaries at clip edges have limited "
        "interpretability; compare models on the identical masked ranges.\n\n"
        "Do not treat unlabelled clips or pseudo-labels as independent ground truth. Store all "
        "annotations and clips privately under artifacts/.\n"
    )
    return plan
