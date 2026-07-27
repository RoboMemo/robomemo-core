"""
pose3d.fuse.temporal — temporal smoothing of joint positions across frames.

Zero-phase (forward-backward) Butterworth low-pass on joint x/y/z, plus
majority-vote smoothing on source labels to prevent flickering.
"""
from __future__ import annotations
from collections import Counter
import numpy as np

try:
    from scipy.signal import butter, sosfiltfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def _butterworth_sos(cutoff_hz: float, fs: float, order: int = 3):
    """Design a second-order-section Butterworth low-pass filter."""
    nyq = fs / 2.0
    # Clamp cutoff to just below Nyquist to avoid numerical issues
    cutoff_hz = min(cutoff_hz, nyq * 0.95)
    return butter(order, cutoff_hz / nyq, btype="low", output="sos")


def smooth_positions(joint_xyz: np.ndarray, sos: np.ndarray) -> np.ndarray:
    """Low-pass filter a (N_frames, 3) position array. Returns filtered array.

    Uses zero-phase filtering (forward + backward) for zero temporal lag.
    If scipy is unavailable, returns the input unchanged (graceful degradation).
    """
    if not HAS_SCIPY or joint_xyz.shape[0] < 4:
        return joint_xyz
    out = np.empty_like(joint_xyz)
    for axis in range(3):
        col = joint_xyz[:, axis]
        if np.all(np.isfinite(col)):
            out[:, axis] = sosfiltfilt(sos, col)
        else:
            # Interpolate NaN/Inf gaps before filtering
            valid = np.isfinite(col)
            if valid.sum() < 4:
                out[:, axis] = col
            else:
                idx = np.arange(len(col))
                col_clean = np.interp(idx, idx[valid], col[valid])
                out[:, axis] = sosfiltfilt(sos, col_clean)
    return out


def smooth_sources(source_list: list[str], window: int = 5) -> list[str]:
    """Majority-vote smoothing on source labels over a sliding window.

    Prevents frame-to-frame flickering between 'triangulated' and
    'singleview'. Returns a new list of the same length.
    """
    n = len(source_list)
    if n == 0 or window <= 1:
        return list(source_list)
    out = []
    half = window // 2
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        neighborhood = source_list[lo:hi]
        # Prefer 'triangulated' over 'singleview' when counts are equal
        counts = Counter(neighborhood)
        best = max(counts, key=lambda k: (counts[k], k == "triangulated"))
        out.append(best)
    return out


def smooth_poses(poses_sequence: list[dict], joint_names: list[str],
                 fps: float = 59.94, cutoff_hz: float = 6.0,
                 order: int = 3, source_window: int = 5) -> list[dict]:
    """Apply temporal smoothing to a sequence of per-frame pose dicts.

    Args:
        poses_sequence: list of per-frame pose dicts (length N).
        joint_names: list of all joint names to process.
        fps: frame rate of the source video.
        cutoff_hz: low-pass cutoff frequency.
        order: Butterworth filter order.
        source_window: sliding window for source label smoothing.
    """
    n_frames = len(poses_sequence)
    if n_frames < 4:
        return poses_sequence

    sos = _butterworth_sos(cutoff_hz, fps, order) if HAS_SCIPY else None

    # Collect per-joint position arrays: (N_frames, 3)
    joint_arrays = {}
    joint_valid = {}
    for name in joint_names:
        arr = np.full((n_frames, 3), np.nan)
        valid = np.zeros(n_frames, dtype=bool)
        for t, frame in enumerate(poses_sequence):
            j = frame.get(name)
            if j is not None:
                xyz = j.get("xyz")
                if xyz is not None and np.all(np.isfinite(xyz)):
                    arr[t] = np.asarray(xyz, float)
                    valid[t] = True
        joint_arrays[name] = arr
        joint_valid[name] = valid

    # Smooth positions
    smoothed_arrays = {}
    for name in joint_names:
        if sos is not None:
            smoothed_arrays[name] = smooth_positions(joint_arrays[name], sos)
        else:
            smoothed_arrays[name] = joint_arrays[name]

    # Collect per-joint source arrays
    source_arrays = {}
    for name in joint_names:
        src = ["missing"] * n_frames
        for t, frame in enumerate(poses_sequence):
            j = frame.get(name)
            if j is not None:
                src[t] = j.get("source", "missing")
        source_arrays[name] = src

    # Smooth source labels
    smoothed_sources = {}
    for name in joint_names:
        smoothed_sources[name] = smooth_sources(source_arrays[name], source_window)

    # Reassemble into pose dicts
    out = []
    for t in range(n_frames):
        frame = {}
        for name in joint_names:
            orig = poses_sequence[t].get(name, {})
            xyz = smoothed_arrays[name][t]
            # Restore None for joints that were originally missing
            if not joint_valid[name][t]:
                xyz = orig.get("xyz")
            frame[name] = {
                "xyz": xyz,
                "conf": orig.get("conf", 0.0),
                "source": smoothed_sources[name][t],
                "used_views": orig.get("used_views", []),
            }
        out.append(frame)
    return out
