"""Observation and phase evidence, independent of public or private labels."""

import numpy as np
from scipy.ndimage import uniform_filter1d

from possession.contracts import positive_fps, validate_ranges


def smooth(values, seconds, fps, ranges):
    """Centered offline mean; never mix halves. Edge replication preserves constants."""
    positive_fps(fps)
    if not np.isfinite(seconds) or seconds < 0:
        raise ValueError("Smoothing duration must be finite and non-negative")
    ranges = validate_ranges(ranges, len(values))
    size = min(max(1, int(round(seconds * fps)) | 1), max(1, 2 * len(values) + 1))
    out = np.zeros_like(values, dtype=np.float64)
    for start, stop in ranges:
        out[start:stop] = uniform_filter1d(
            np.asarray(values[start:stop], dtype=float), size=size, axis=0, mode="nearest"
        )
    return out


def missing_age(observed, fps, ranges):
    """Distance to the nearest observation, and total gap length, both in seconds."""
    age = np.zeros(len(observed), dtype=float)
    gap = np.zeros(len(observed), dtype=float)
    for start, stop in ranges:
        missing = ~observed[start:stop]
        edges = np.diff(np.r_[False, missing, False].astype(np.int8))
        for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1), strict=True):
            t = np.arange(a, b)
            left = t - a + 1 if a else np.full(b - a, b - a)
            right = b - t if b < stop - start else np.full(b - a, b - a)
            age[start + a : start + b] = np.minimum(left, right) / fps
            gap[start + a : start + b] = (b - a) / fps
    return age, gap


ATTACK = {
    "BUILD UP": 1.0,
    "PROGRESSION": 1.0,
    "CREATING CHANCES": 1.0,
    "ATTACKING TRANSITION": 1.0,
    "LONG BALL": 0.7,
    "BALL IN FINAL THIRD": 0.65,
    "BALL IN THE BOX": 0.65,
    "PLAYERS IN THE BOX": 0.3,
}
DEFENSE = {
    "DEFENDING IN ATTACKING THIRD",
    "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD",
    "DEFENSIVE TRANSITION",
}
TEAMS = {"barcelona": 0, "atletic club": 1, "athletic club": 1}


def tag_features(events, n, fps, ranges):
    team_support = np.zeros((n, 2), dtype=float)
    live = np.zeros(n, dtype=float)
    set_piece = np.zeros(n, dtype=float)
    restart_team = np.zeros((n, 2), dtype=float)
    for event in events:
        a, b = max(0, event["start_frame"]), min(n, event["end_frame"])
        if b <= a:
            continue
        code = (event["code"] or "").strip().upper()
        team = TEAMS.get(event["tags"].get("Team", "").strip().lower())
        if code == "SET PIECES":
            set_piece[a:b] = 1
            if team is not None:
                restart_team[a:b, team] = 1
            continue
        weight = ATTACK.get(code, 0.0)
        if code in DEFENSE:
            weight = 1.0
            team = 1 - team if team is not None else None
        if weight:
            # Related tags are correlated: use max, not a growing vote count.
            live[a:b] = np.maximum(live[a:b], weight)
            if team is not None:
                team_support[a:b, team] = np.maximum(team_support[a:b, team], weight)
    team_support = smooth(team_support, 0.6, fps, ranges)
    return {
        "tag_home": team_support[:, 0],
        "tag_away": team_support[:, 1],
        "tag_live": smooth(live, 0.6, fps, ranges),
        "tag_set_piece": smooth(set_piece, 0.4, fps, ranges),
        "tag_restart_home": smooth(restart_team[:, 0], 0.4, fps, ranges),
        "tag_restart_away": smooth(restart_team[:, 1], 0.4, fps, ranges),
    }


def build_features(players, ball, team, events, pitch, fps, ranges):
    n = len(ball)
    positive_fps(fps)
    ranges = validate_ranges(ranges, n)
    ball_xy = ball[:, :2].astype(float) * np.array(pitch)
    player_xy = players[:, :, :2].astype(float) * np.array(pitch)
    observed = np.isfinite(ball_xy).all(axis=1)
    player_valid = np.isfinite(player_xy).all(axis=2)
    displacement_speed = np.full(n, np.nan)
    # Central difference is not used across missing samples, track discontinuities or halves.
    for start, stop in ranges:
        speed = np.linalg.norm(np.diff(ball_xy[start:stop], axis=0), axis=1) * fps
        displacement_speed[start + 1 : stop] = speed
    # A very large jump invalidates both endpoints; the model should not pick a side of it.
    jump = np.nan_to_num(displacement_speed, nan=0) > 70.0
    unreliable = jump.copy()
    for start, stop in ranges:
        unreliable[start : stop - 1] |= jump[start + 1 : stop]
    in_bounds = (
        (ball_xy[:, 0] >= -5)
        & (ball_xy[:, 0] <= pitch[0] + 5)
        & (ball_xy[:, 1] >= -5)
        & (ball_xy[:, 1] <= pitch[1] + 5)
    )
    reliable = observed & ~unreliable & in_bounds
    speed = np.clip(np.nan_to_num(displacement_speed, nan=0), 0, 40)
    # Geometry is aggregated by team, without treating track slots as persistent identities.
    distances = np.linalg.norm(player_xy - ball_xy[:, None, :], axis=2)
    distances[~player_valid | ~reliable[:, None]] = np.inf
    nearest = np.stack([np.min(distances[:, team == t], axis=1) for t in (0, 1)], axis=1)
    nearest = np.minimum(nearest, 100.0)
    closest = nearest.min(axis=1)
    both_teams = np.stack([player_valid[:, team == t].any(axis=1) for t in (0, 1)], axis=1)
    contact = np.exp(-0.5 * (closest / 4.0) ** 2) * reliable * both_teams.all(axis=1)
    advantage = np.tanh((nearest[:, 1] - nearest[:, 0]) / 2.0)
    control = advantage * contact
    # A pass can travel near a defender without being controlled by that player. Compare
    # motion over 0.4 seconds, requiring continuous ball observations and plausible slot motion.
    motion_factor = np.ones_like(distances)
    half_window = max(1, round(0.2 * fps))
    for start, stop in ranges:
        centre = np.arange(start + half_window, stop - half_window)
        if not len(centre):
            continue
        left, right = centre - half_window, centre + half_window
        time_s = 2 * half_window / fps
        count_prefix = np.r_[0, np.cumsum(reliable)]
        continuous = count_prefix[right + 1] - count_prefix[left] == 2 * half_window + 1
        ball_velocity = (ball_xy[right] - ball_xy[left]) / time_s
        player_velocity = (player_xy[right] - player_xy[left]) / time_s
        plausible = (
            player_valid[left]
            & player_valid[right]
            & (np.linalg.norm(player_velocity, axis=2) <= 12)
        )
        usable = continuous[:, None] & plausible
        relative_speed = np.linalg.norm(player_velocity - ball_velocity[:, None, :], axis=2)
        compatibility = 0.25 + 0.75 * np.exp(-0.5 * (relative_speed / 8.0) ** 2)
        motion_factor[centre] = np.where(usable, compatibility, 1.0)
    player_control = np.exp(-0.5 * (distances / 4.0) ** 2) * motion_factor
    support = np.stack([player_control[:, team == t].max(axis=1) for t in (0, 1)], axis=1)
    balance = (support[:, 0] - support[:, 1]) / np.maximum(support.sum(axis=1), 1e-12)
    motion_control = np.tanh(3 * balance) * support.max(axis=1) * both_teams.all(axis=1)
    speeds = players[:, :, 2].astype(float)
    # Supplied speed avoids differentiating across slot identity switches.
    valid_speed = player_valid & np.isfinite(speeds) & (speeds >= 0) & (speeds <= 12)
    count = player_valid.sum(axis=1)
    moving = ((speeds > 2.0) & valid_speed).sum(axis=1) / np.maximum(valid_speed.sum(axis=1), 1)
    mean_speed = np.where(valid_speed, speeds, 0).sum(axis=1) / np.maximum(
        valid_speed.sum(axis=1), 1
    )
    edge = np.minimum.reduce(
        [ball_xy[:, 0], pitch[0] - ball_xy[:, 0], ball_xy[:, 1], pitch[1] - ball_xy[:, 1]]
    )
    age, gap = missing_age(reliable, fps, ranges)
    values = {
        "ball_observed": observed,
        "ball_reliable": reliable,
        "ball_speed": speed * reliable,
        "ball_moving": (speed > 2.0) & reliable,
        "ball_still": (speed < 0.5) & reliable & np.isfinite(displacement_speed),
        "ball_outside": (edge < -0.3) & reliable,
        "ball_near_edge": (edge < 1.5) & reliable,
        "nearest_home": nearest[:, 0],
        "nearest_away": nearest[:, 1],
        "contact": contact,
        "control": control,
        "motion_control": motion_control,
        "player_count": count,
        "moving_fraction": moving,
        "player_speed": mean_speed,
        "missing_age": age,
        "missing_gap": gap,
    }
    values.update(tag_features(events, n, fps, ranges))
    return {
        key: value.astype(bool if value.dtype == bool else np.float32)
        for key, value in values.items()
    }
