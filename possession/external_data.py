"""Pinned IDSSE ingestion. Competition data and competition labels are never inputs."""

import argparse
import hashlib
import json
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from possession.data import sha256, write_json, write_npz

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "protocols/idsse_v1.json"
SOURCES = ROOT / "protocols/idsse_sources_v1.json"
DATA = ROOT / "artifacts/external_idsse"
PERIODS = {"firstHalf": 1, "secondHalf": 2}


def protocol():
    return json.loads(PROTOCOL.read_text())


def source_records(match):
    spec = protocol()
    allowed = sum(spec["splits"].values(), [])
    if match not in allowed or len(allowed) != len(set(allowed)):
        raise ValueError("Unknown external match or overlapping match splits")
    records = [r for r in json.loads(SOURCES.read_text())["files"] if r["match"] == match]
    if {r["kind"] for r in records} != {"meta", "tracking"} or len(records) != 2:
        raise ValueError("Expected exactly one metadata and one tracking source")
    return records


def verify_source(path, record):
    if path.stat().st_size != record["size"]:
        raise ValueError("External source size mismatch")
    if record["sha256"]:
        valid = sha256(path) == record["sha256"]
    else:
        content = path.read_bytes()
        valid = hashlib.sha1(b"blob " + str(len(content)).encode() + b"\0" + content).hexdigest()
        valid = valid == record["git_blob_sha1"]
    if not valid:
        raise ValueError("External source fingerprint mismatch")


def download(match, root=DATA):
    root.mkdir(parents=True, exist_ok=True)
    for record in source_records(match):
        target = root / record["filename"]
        if target.exists():
            verify_source(target, record)
            continue
        temporary = target.with_suffix(".partial")
        with (
            urllib.request.urlopen(record["url"], timeout=120) as response,
            temporary.open("wb") as f,
        ):
            while chunk := response.read(1024 * 1024):
                f.write(chunk)
        verify_source(temporary, record)
        temporary.replace(target)


def metadata(path, match):
    root = ET.parse(path).getroot()
    general = root.find(".//General")
    if general is None or general.get("MatchId") != f"DFL-MAT-{match}":
        raise ValueError("Metadata match identity differs from pinned source")
    if general.get("Season") != "2022/2023":
        raise ValueError("Unexpected external season")
    environment = root.find(".//Environment")
    pitch = (float(environment.get("PitchX")), float(environment.get("PitchY")))
    if not all(0 < x < 200 for x in pitch):
        raise ValueError("Invalid pitch dimensions")
    teams, players = {}, {}
    for team in root.iter("Team"):
        role = team.get("Role")
        if role not in ("home", "guest"):
            raise ValueError("Unknown team role")
        side = 0 if role == "home" else 1
        if side in teams.values() or team.get("TeamId") in teams:
            raise ValueError("Duplicate team role or identity")
        teams[team.get("TeamId")] = side
        for p in team.iter("Player"):
            pid = p.get("PersonId")
            if not pid or pid in players:
                raise ValueError("Missing or duplicate player identity")
            players[pid] = (len(players), side, team.get("TeamId"))
    if set(teams.values()) != {0, 1}:
        raise ValueError("Both home and away teams are required")
    return pitch, teams, players


def frames(path, match):
    """Stream one object frame at a time; never materialise the full XML tree."""
    current = None
    for event, element in ET.iterparse(path, events=("start", "end")):
        if event == "start" and element.tag == "FrameSet":
            if element.get("MatchId") != f"DFL-MAT-{match}":
                raise ValueError("Tracking match identity differs from metadata")
            period = PERIODS.get(element.get("GameSection"))
            if period is None:
                raise ValueError("Unsupported external period")
            current = (period, element.get("TeamId"), element.get("PersonId"))
        elif event == "end" and element.tag == "Frame":
            if current is None:
                raise ValueError("Frame outside a FrameSet")
            yield (*current, dict(element.attrib))
            element.clear()
        elif event == "end" and element.tag == "FrameSet":
            current = None
            element.clear()


def label(status, possession):
    if status == "0":
        return 2
    if status == "1" and possession in ("1", "2"):
        return int(possession) - 1
    return 3  # Missing/unknown ownership must not silently become away possession.


def load_arrays(meta_path, tracking_path, match):
    pitch, teams, roster = metadata(meta_path, match)
    ball_rows = {1: {}, 2: {}}
    for period, tid, _, row in frames(tracking_path, match):
        if tid == "BALL":
            number = int(row["N"])
            if number in ball_rows[period]:
                raise ValueError("Duplicate ball frame")
            ball_rows[period][number] = row
    ranges, bounds, total = [], {}, 0
    for period, rows in ball_rows.items():
        if not rows:
            raise ValueError("Both complete halves are required")
        lo, hi = min(rows), max(rows) + 1
        if not 0 < hi - lo < 200000:
            raise ValueError("Invalid external half length")
        bounds[period] = (lo, hi, total)
        ranges.append((total, total + hi - lo))
        total += hi - lo
    players = np.full((total, len(roster), 3), np.nan, dtype=np.float32)
    ball = np.full((total, 2), np.nan, dtype=np.float32)
    truth = np.full(total, 3, dtype=np.int8)
    for period, rows in ball_rows.items():
        lo, _, offset = bounds[period]
        for number, row in rows.items():
            i = offset + number - lo
            ball[i] = [float(row.get(k, "nan")) for k in ("X", "Y")]
            truth[i] = label(row.get("BallStatus"), row.get("BallPossession"))
    del ball_rows
    seen = np.zeros((total, len(roster)), dtype=bool)
    for period, tid, pid, row in frames(tracking_path, match):
        if tid not in teams:
            continue  # Ball and officials do not become player features.
        if pid not in roster or roster[pid][2] != tid:
            raise ValueError("Player roster/team mismatch")
        lo, hi, offset = bounds[period]
        number = int(row["N"])
        if not lo <= number < hi:
            continue
        slot = roster[pid][0]
        key = (offset + number - lo, slot)
        if seen[key]:
            raise ValueError("Duplicate player frame")
        seen[key] = True
        players[offset + number - lo, slot] = (
            float(row.get("X", "nan")),
            float(row.get("Y", "nan")),
            float(row.get("S", "nan")) / 3.6,  # Provider specifies km/h.
        )
    sides = np.array([value[1] for value in roster.values()], dtype=np.int8)
    return players, ball, sides, pitch, ranges, truth


def dropout_mask(n, ranges, match):
    spec = protocol()
    seed = spec["dropout"]["seed"] + int(hashlib.sha256(match.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    mask = np.zeros(n, dtype=bool)
    size = round(spec["fps"] * spec["dropout"]["block_seconds"])
    for a, b in ranges:
        for start in range(a, b, size):
            if rng.random() < spec["dropout"]["fraction"]:
                mask[start : min(start + size, b)] = True
    return mask


def prepare_match(match, root=DATA):
    from possession.external_model import observation_features, reference_labels

    records = source_records(match)
    paths = {}
    for record in records:
        p = root / record["filename"]
        verify_source(p, record)
        paths[record["kind"]] = p
    destination = root / f"{match}.npz"
    receipt_path = root / f"{match}.cache.json"
    identity = {
        "protocol_sha256": sha256(PROTOCOL),
        "sources_sha256": sha256(SOURCES),
        "ingestion_sha256": sha256(__file__),
        "feature_code_sha256": sha256(Path(__file__).with_name("external_model.py")),
        "inputs_sha256": {k: sha256(v) for k, v in paths.items()},
    }
    if destination.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt.get("identity") == identity and receipt.get("npz_sha256") == sha256(destination):
            return destination
        raise ValueError("External cache changed; preserve it and rebuild in a new directory")
    players, ball, teams, pitch, ranges, truth = load_arrays(
        paths["meta"], paths["tracking"], match
    )
    arrays = {"truth": truth, "ranges": np.array(ranges)}
    mask = dropout_mask(len(ball), ranges, match)
    for condition in protocol()["conditions"]:
        observed = ball.copy()
        if condition == "ball_dropout":
            observed[mask] = np.nan
        live, team, names = observation_features(players, observed, teams, pitch, ranges)
        arrays[f"live_{condition}"] = live.astype(np.float32)
        arrays[f"team_{condition}"] = team.astype(np.float32)
        arrays[f"reference_{condition}"] = reference_labels(players, observed, teams, ranges)
    write_npz(destination, arrays)
    write_json(
        receipt_path,
        {
            "identity": identity,
            "npz_sha256": sha256(destination),
            "match": match,
            "frames": len(truth),
            "known_labels": int((truth != 3).sum()),
            "label_counts": {str(i): int((truth == i).sum()) for i in range(4)},
            "feature_names": names,
            "masked_frames": int(mask.sum()),
        },
    )
    print(
        f"Prepared external match {match}: {len(truth)} frames, {(truth != 3).mean():.1%} labelled",
        flush=True,
    )
    return destination


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["download", "prepare"])
    parser.add_argument("--root", type=Path, default=DATA)
    args = parser.parse_args()
    spec = protocol()
    matches = spec["splits"]["train"] + spec["splits"]["development"]
    if args.command == "download":
        for match in matches + spec["splits"]["holdout"]:
            download(match, args.root)
    else:
        for match in matches:
            prepare_match(match, args.root)


if __name__ == "__main__":
    main()
