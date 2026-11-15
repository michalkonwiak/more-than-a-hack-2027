"""Shared numerical and timeline contracts, including synthetic test timelines."""

import numpy as np


def positive_fps(fps):
    if (
        isinstance(fps, (bool, np.bool_))
        or not isinstance(fps, (int, float, np.integer, np.floating))
        or not np.isfinite(fps)
        or fps <= 0
    ):
        raise ValueError("fps must be a positive finite number")


def validate_ranges(ranges, n):
    out = list(ranges)
    previous = 0
    for start, stop in out:
        if any(
            isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
            for v in (start, stop)
        ):
            raise ValueError("Frame ranges must contain integer boundaries")
        if not 0 <= start < stop <= n or start < previous:
            raise ValueError("Frame ranges must be sorted, non-overlapping and inside the timeline")
        previous = stop
    return out


def label_array(labels, n):
    labels = np.asarray(labels)
    if labels.shape != (n,) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"Expected {n} integer H/A/D/X codes")
    if not np.isin(labels, [0, 1, 2, 3]).all():
        raise ValueError("Labels contain a code outside H/A/D/X")
    return labels
