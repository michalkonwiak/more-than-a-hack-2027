"""Strict tactical ingestion and content-addressed, local-only feature caches."""

import hashlib
import importlib.metadata
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from games import GAME, GAMES, asset_paths, half_video_blocks
from starter.baseline import team_of_each_track
from starter.load_smart_tagging import check_time_base, load


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def write_npz(path, arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


@contextmanager
def cache_lock(path):
    """Serialise same-key builders. A process exit automatically releases the advisory lock."""
    import fcntl

    with Path(path).open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def read_cache(path, meta_path, key):
    """A cache hit must have the expected key, a matching digest and matching array lengths."""
    report = json.loads(Path(meta_path).read_text())
    if report.get("cache_key") != key or report.get("archive_sha256") != sha256(path):
        raise ValueError("Feature cache integrity check failed; remove this cache pair to rebuild")
    with np.load(path, allow_pickle=False) as stored:
        features = {name: stored[name] for name in stored.files}
    n = report["n_frames"]
    if not features or any(value.shape != (n,) for value in features.values()):
        raise ValueError("Feature cache array dimensions do not match its manifest")
    return features, report


def blocks():
    """Half-open video ranges. Do NOT apply legacy corrected-IdFrame offsets to ATD."""
    return [(start, end + 1) for _, start, end in half_video_blocks(GAME)]


def graded_mask(n):
    out = np.zeros(n, dtype=bool)
    for start, stop in blocks():
        out[start:stop] = True
    return out


def read_tracking(path, n_frames, n_tracks):
    """Reject truncated, duplicate, reordered or malformed exports instead of repairing them."""
    players = np.full((n_frames, n_tracks, 3), np.nan, dtype=np.float32)
    ball = np.full((n_frames, 3), np.nan, dtype=np.float32)
    count = 0
    with Path(path).open() as stream:
        for count, line in enumerate(stream, 1):
            fields = line.rstrip().split(":")
            if count > n_frames or len(fields) != 3 or int(fields[0]) != count:
                raise ValueError(f"Invalid tracking frame/index at record {count}")
            groups = fields[1].rstrip(";").split(";")
            if len(groups) != n_tracks:
                raise ValueError(f"Expected {n_tracks} player slots at record {count}")
            if any(group.count(",") != 2 for group in groups):
                raise ValueError(f"Each player slot needs x,y,speed at record {count}")
            values = np.fromstring(fields[1].rstrip(";").replace(";", ","), sep=",")
            observed_ball = np.fromstring(fields[2].rstrip(";"), sep=",")
            if values.size != n_tracks * 3 or observed_ball.size != 3:
                raise ValueError(f"Malformed coordinates at record {count}")
            players[count - 1] = values.reshape(n_tracks, 3)
            ball[count - 1] = observed_ball
    if count != n_frames:
        raise ValueError(f"Tracking has {count} records; expected {n_frames}")
    return players, ball


def input_manifest(data_dir):
    paths = asset_paths(data_dir=str(data_dir))
    # Video is not consumed by this model and does not belong in its cache key.
    assets = {k: Path(v) for k, v in paths.items() if k != "video"}
    for path in assets.values():
        if not path.is_file():
            raise FileNotFoundError(f"Missing asset: {path}; set --data-dir or MTAH_DATA_DIR")
    return assets, {
        k: {"sha256": sha256(p), "bytes": p.stat().st_size} for k, p in sorted(assets.items())
    }


def prepare(data_dir, cache_dir):
    assets, manifest = input_manifest(data_dir)
    # Include code and competition constants, so a feature fix cannot reuse stale values.
    code_files = [
        Path(__file__),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("contracts.py"),
        Path(__file__).parents[1] / "games.py",
        Path(__file__).parents[1] / "starter/load_smart_tagging.py",
        Path(__file__).parents[1] / "starter/baseline.py",
    ]
    key_payload = {
        "inputs": manifest,
        "code": {p.name: sha256(p) for p in code_files},
        "dependencies": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy")},
    }
    key = hashlib.sha256(json.dumps(key_payload, sort_keys=True).encode()).hexdigest()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"features-{key}.npz"
    meta_path = path.with_suffix(".json")
    with cache_lock(path.with_suffix(".lock")):
        if path.exists() and meta_path.exists():
            features, report = read_cache(path, meta_path, key)
            return features, report, path
        return _prepare_uncached(assets, manifest, data_dir, path, meta_path, key, key_payload)


def _prepare_uncached(assets, manifest, data_dir, path, meta_path, key, key_payload):
    from possession.features import build_features

    n = GAMES[GAME]["n_video_frames"]
    team = team_of_each_track(assets["atd_meta"])
    players, ball = read_tracking(assets["atd_raw"], n, len(team))
    homography = pd.read_csv(assets["homography"])
    if not np.array_equal(homography["frame"].to_numpy(), np.arange(n)):
        raise ValueError("Homography must contain exactly the tactical video frame indices")
    matrix = homography[[f"h{i}{j}" for i in range(3) for j in range(3)]].to_numpy()
    valid = homography["valid"].to_numpy()
    if not np.isin(valid, [0, 1]).all():
        raise ValueError("Invalid homography validity flags")
    if not np.isfinite(matrix[valid == 1]).all():
        raise ValueError("Homography marked valid contains non-finite matrix entries")
    error = homography["reprojection_error"].to_numpy(np.float32)
    if not (np.isfinite(error[valid == 1]) & (error[valid == 1] >= 0)).all():
        raise ValueError("Valid homography must have finite, non-negative reprojection error")
    events = load(data_dir=str(data_dir))
    if not check_time_base(events)["agrees"]:
        raise ValueError("Smart Tagging clock disagrees with the published kickoff anchors")
    features = build_features(
        players, ball, team, events, GAMES[GAME]["pitch"], GAMES[GAME]["fps"], blocks()
    )
    features["homography_error"] = homography["reprojection_error"].to_numpy(np.float32)
    features["homography_valid"] = valid.astype(bool)
    graded = graded_mask(n)
    report = {
        "cache_key": key,
        "cache_recipe": key_payload,
        "inputs": manifest,
        "n_frames": n,
        "n_tracks": len(team),
        "n_events": len(events),
        "n_graded": int(graded.sum()),
        "ball_observed_fraction": float(features["ball_observed"].mean()),
        "ball_observed_graded_fraction": float(features["ball_observed"][graded].mean()),
        "ball_reliable_graded_fraction": float(features["ball_reliable"][graded].mean()),
        "homography_valid_fraction": float(valid.mean()),
        "mean_tracked_players_graded": float(features["player_count"][graded].mean()),
        "tag_live_coverage_graded": float((features["tag_live"][graded] > 0).mean()),
    }
    # Detect input changes during preparation before publishing the cache pair.
    if any(sha256(asset) != manifest[name]["sha256"] for name, asset in assets.items()):
        raise ValueError(
            "Input assets changed during feature preparation; retry with stable inputs"
        )
    write_npz(path, features)
    report["archive_sha256"] = sha256(path)
    write_json(meta_path, report)
    return features, report, path
