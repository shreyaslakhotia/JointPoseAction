"""Conservative E0 exclusions; never repair a recording by guessing its clock."""
import json
from pathlib import Path

import numpy as np


EXCLUDED_RECORDINGS = {
    "nusar-2021_action_both_9031-c10b_9031_user_id_2021-02-04_101221": "non_60fps_timestamp_cadence",
    "nusar-2021_action_both_9026-b04b_9026_user_id_2021-02-03_163855": "incomplete_annotation_pose_coverage",
}


def load_confidences(path):
    """Observed official companion schema: frame -> hand key -> scalar score."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("Expected nonempty frame-keyed confidence object")
    result = {}
    for key, row in value.items():
        if not key.isdigit() or int(key) in result or not isinstance(row, dict) or set(row) != {"0", "1"}:
            raise ValueError("Unexpected confidence frame/hand keys")
        if any(not isinstance(x, (int, float)) or isinstance(x, bool) for x in row.values()):
            raise ValueError("Scalar confidence scores required")
        scores = np.asarray([row["0"], row["1"]], dtype=float)
        if not np.isfinite(scores).all() or (scores < 0).any() or (scores > 1).any():
            raise ValueError("Confidence scores must be finite in [0,1]")
        result[int(key)] = scores
    return result


def timestamp_cadence(path, frames, fps=60, tolerance_seconds=0.002):
    """Check observed seconds-like cadence; does not establish video clock offset."""
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not value:
        raise ValueError("Expected nonempty frame-keyed timestamps")
    times = {}
    for key, stamp in value.items():
        if not key.isdigit() or int(key) in times or isinstance(stamp, bool) or not isinstance(stamp, (int, float)):
            raise ValueError("Unexpected timestamp schema")
        if not np.isfinite(stamp):
            raise ValueError("Nonfinite timestamp")
        times[int(key)] = stamp
    if set(times) != set(frames) or len(times) < 2:
        return {"accepted": False, "reason": "timestamp_pose_keys_mismatch_or_too_short"}
    keys = sorted(times)
    residual = np.array([(times[b] - times[a]) - (b - a) / fps for a, b in zip(keys, keys[1:])])
    accepted = bool((np.abs(residual) <= tolerance_seconds).all())
    return {"accepted": accepted, "reason": None if accepted else "non_60fps_timestamp_cadence",
            "maximum_step_residual_seconds": float(np.abs(residual).max()),
            "tolerance_seconds": tolerance_seconds, "video_clock_offset_verified": False}
