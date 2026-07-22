"""
pose3d.calib.multicam_calib — 3-camera calibration from a checkerboard recording.

Pipeline:
  1. Sample frames from each (synchronized) calibration video at sample_fps.
  2. Detect the board (checkerboard.detect, pattern (11,8) -> fallback (10,7)).
  3. Per-camera intrinsics K/dist (cv2.calibrateCamera, METRIC object points,
     square_size = config board_square_size_m = 0.02 m).
  4. Pairwise stereoCalibrate(H,L) & (H,R) with CALIB_FIX_INTRINSIC -> rigid
     relative extrinsics R_{HL},t_{HL}, R_{HR},t_{HR} (meters). These are the
     constant relative transforms between the 3 head cameras (rigid rig).

The output is the single source of metric scale for the whole pipeline.

ASSUMPTION: the 3 calibration videos are temporally synchronized (pair by
sampled frame index). run_pipeline audio-aligns the cali recording first; if
your cali videos have no audio, record them synchronously or align manually.
"""
from __future__ import annotations
import json
import numpy as np
import cv2

from .checkerboard import detect, object_points
from ..io.video_reader import VideoReader


def _detect_in_video(path: str, primary, fallback, square_size_m, sample_fps):
    """Return list of (sample_idx, cornersNx2, pattern) for detected frames."""
    found = []
    with VideoReader(path) as r:
        for sidx, _t, rgb in r.sample_at_fps(sample_fps):
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
            corners, pat = detect(gray, primary, fallback)
            if corners is not None:
                found.append((sidx, corners, pat))
    return found


def _align_pattern(found_per_view):
    """Keep only the pattern used by the majority view; drop minority detections.

    The board pattern (11,8 vs 10,7) must be identical across views for a shared
    object-point frame. If views disagree, keep the most common one.
    """
    from collections import Counter
    cnt = Counter(p for lst in found_per_view for (_, _, p) in lst)
    if not cnt:
        return found_per_view, None
    keep_pat = cnt.most_common(1)[0][0]
    out = []
    for lst in found_per_view:
        out.append([(s, c, p) for (s, c, p) in lst if p == keep_pat])
    return out, keep_pat


def calibrate_multicam(cali_paths: dict, cfg: dict) -> dict:
    """cali_paths: {view: video_path}. cfg: the 'calibration' sub-config.

    Returns calibration dict (also JSON-serializable):
      {views, K{view}, dist{view}, extrinsics{view:{R,t}}, board_square_size_m,
       img_size, pattern_used, rms{view,HL,HR}}
    """
    views = list(cali_paths.keys())
    primary = (cfg["board_cols"], cfg["board_rows"])
    fallback = tuple(cfg["pattern_fallback"])
    sq = float(cfg["board_square_size_m"])
    sample_fps = float(cfg["sample_fps"])

    # 1. detect board per view
    found = {v: _detect_in_video(cali_paths[v], primary, fallback, sq, sample_fps)
             for v in views}
    found_lists, keep_pat = _align_pattern([found[v] for v in views])
    found = {v: found_lists[i] for i, v in enumerate(views)}
    if keep_pat is None:
        raise RuntimeError("No checkerboard detected in any calibration video. "
                           "Check board dims / pattern / focus.")

    objp = object_points(keep_pat, sq)

    # per-view image size — cameras may differ (H 1920x1440 landscape,
    # L/R 1440x1920 portrait). Each camera MUST be calibrated with its OWN size.
    img_size = {}
    for v in views:
        with VideoReader(cali_paths[v]) as r:
            img_size[v] = (r.width, r.height)

    # 2. per-view intrinsics
    K, dist, rms = {}, {}, {}
    for v in views:
        obj_pts, img_pts = [], []
        for _sidx, corners, _p in found[v]:
            obj_pts.append(objp)
            img_pts.append(corners.reshape(-1, 1, 2).astype(np.float32))
        if len(obj_pts) < 6:
            raise RuntimeError(f"View {v}: only {len(obj_pts)} board detections "
                               f"(need >=6). Lower sample_fps or re-record cali.")
        ret, K_, dist_, _, _ = cv2.calibrateCamera(obj_pts, img_pts, img_size[v], None, None)
        K[v], dist[v], rms[v] = K_, np.ravel(dist_), float(ret)

    # 3. pairwise relative extrinsics (H as reference)
    extrinsics = {views[0]: {"R": np.eye(3), "t": np.zeros(3)}}
    pair_rms = {}
    if len(views) >= 2:
        ref = views[0]
        det = {v: {s: c for (s, c, _p) in found[v]} for v in views}
        for other in views[1:]:
            shared = sorted(set(det[ref].keys()) & set(det[other].keys()))
            obj_pts, pts_ref, pts_o = [], [], []
            for s in shared:
                obj_pts.append(objp)
                pts_ref.append(det[ref][s].reshape(-1, 1, 2).astype(np.float32))
                pts_o.append(det[other][s].reshape(-1, 1, 2).astype(np.float32))
            if len(obj_pts) < int(cfg.get("min_shared_frames", 8)):
                raise RuntimeError(f"Stereo {ref}-{other}: only {len(obj_pts)} shared "
                                   f"detections. Re-record cali synchronously.")
            ret, R, t, K_o, d_o, cfg_name = _stereo_best(
                obj_pts, pts_ref, pts_o, K[ref], dist[ref], K[other], dist[other],
                img_size[ref],
                max_baseline_m=float(cfg.get("stereo_max_baseline_m", 0.5)))
            K[other], dist[other] = K_o, d_o          # propagate refined intrinsic
            extrinsics[other] = {"R": R, "t": t.ravel()}
            pair_rms[f"{ref}{other}"] = float(ret)
            rms_thr = float(cfg.get("stereo_rms_thr_px", 1.0))
            if ret > rms_thr:
                print(f"  ⚠ stereo {ref}-{other} RMS={ret:.3f}px > {rms_thr:.1f} "
                      f"— bad board coverage or remaining orientation issue; "
                      f"consider re-recording cali with better overlap.")

    calib = {
        "views": views,
        "img_size": img_size,
        "pattern_used": list(keep_pat),
        "board_square_size_m": sq,
        "K": {v: K[v].tolist() for v in views},
        "dist": {v: dist[v].tolist() for v in views},
        "extrinsics": {v: {"R": extrinsics[v]["R"].tolist(),
                            "t": extrinsics[v]["t"].tolist()} for v in views},
        "rms": {"intrinsics": {v: rms[v] for v in views}, **pair_rms},
    }
    return calib


def _stereo_best(obj_pts, pts_ref, pts_other, K1, d1, K2, d2, img_size,
                 max_baseline_m=0.5):
    """Pick the best stereoCalibrate solution across flag-configs x 180-flip,
    penalizing unrealistic baselines.

    With weakly-constrained data (limited board coverage), different flag configs
    converge to wildly different baselines (e.g. 17cm vs 261cm for the same pair).
    A physically-valid baseline is a strong sanity signal on a head-cam rig
    (baseline << max_baseline_m). Returns (rms, R, t, K2_refined, d2_refined, tag).
    """
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-6)
    configs = {
        "FIX_INTRINSIC": cv2.CALIB_FIX_INTRINSIC,
        "SAME_FOCAL": cv2.CALIB_SAME_FOCAL_LENGTH,
        "JOINT": 0,
    }
    flips = {"asis": pts_other,
             "flip": [p[::-1].copy() for p in pts_other]}
    best = None  # (score, ret, R, t, K2, d2, tag)
    for cname, fl in configs.items():
        for fname, pts_o in flips.items():
            try:
                ret, _K1, _d1, _K2, _d2, R, t, _, _ = cv2.stereoCalibrate(
                    obj_pts, pts_ref, pts_o,
                    K1.copy(), d1.copy(), K2.copy(), d2.copy(),
                    img_size, flags=fl, criteria=crit)
            except cv2.error:
                continue
            if not np.isfinite(ret):
                continue
            bl = float(np.linalg.norm(t))
            score = ret if bl < max_baseline_m else ret + 1e6   # reject absurd baselines
            tag = f"{cname}/{fname}"
            if best is None or score < best[0]:
                best = (score, float(ret), R, t, _K2, np.ravel(_d2), tag)
    if best is None:
        raise RuntimeError("stereoCalibrate failed for all flag/flip configs.")
    _score, ret, R, t, K2r, d2r, tag = best
    return ret, R, t, K2r, d2r, tag


def build_projection_matrices(calib: dict) -> dict:
    """{view: 3x4 ndarray P = K [R|t]} for triangulation (H reference frame).

    IMPORTANT: cv2.stereoCalibrate(ref=H, other=v) returns R,t that map a point
    FROM v's frame TO H's frame (X_H = R·X_v + t). Projection of a world(H)
    point into v's image needs the INVERSE (X_v = R^T·X_H - R^T·t). So we invert
    here. The reference (H) has R=I, t=0, whose inverse is itself -> safe for all.
    """
    Ps = {}
    for v in calib["views"]:
        K = np.asarray(calib["K"][v], dtype=np.float64)
        R = np.asarray(calib["extrinsics"][v]["R"], dtype=np.float64)
        t = np.asarray(calib["extrinsics"][v]["t"], dtype=np.float64).reshape(3, 1)
        R_inv = R.T
        t_inv = -R.T @ t
        Ps[v] = K @ np.hstack([R_inv, t_inv])
    return Ps


def save_calibration(calib: dict, path: str):
    with open(path, "w") as f:
        json.dump(calib, f, indent=2)


def load_calibration(path: str) -> dict:
    with open(path) as f:
        return json.load(f)
