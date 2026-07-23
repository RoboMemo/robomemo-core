#!/usr/bin/env python3
"""
run_pipeline.py — 3D whole-body pose pipeline orchestrator.

    python run_pipeline.py --recording 0721-1
    python run_pipeline.py --recording 0721-1 --no-align   # reuse existing *_aligned.MP4
    python run_pipeline.py --recording 0721-1 --device cuda --body-model h32

Flow:
  1. Time-align the operating recording (reuses Data_Preprocessing/align_audio.py).
  2. Multi-view calibrate from the cali recording (checkerboard 11x8, 20mm) if no
     calibration.json yet. -> metric 3D anchor + projection matrices.
  3. Per frame (lockstep across aligned H/L/R): run SMPLer-X-H32 per view.
  4. Fuse: DLT-triangulate body (triangulated), beta-scale single-view fallback;
     attach hands (single-view, snapped to triangulated wrist).
  5. Write poses.json (78 joints, metric, H frame).
  6. Save sample viz (2D projection overlay = SMPLer-X convention check + 3D body).

Run from this directory (PoseEstimation/pose3d/) so the `pose3d` package imports.
Target: Linux + NVIDIA CUDA. (The --no-calib/--no-align CPU smoke path can run on
the dev Mac for the alignment step only; 3D inference needs CUDA.)
"""
from __future__ import annotations
import os
import sys
import argparse
import subprocess
from collections import Counter

import cv2
import numpy as np
import tqdm
import yaml

# Ensure the inner `pose3d` package is importable when run from this dir.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pose3d.io.time_align import align_recording
from pose3d.io.video_reader import VideoReader
from pose3d.io.pose_writer import write_poses, make_frame
from pose3d.calib.multicam_calib import calibrate_multicam, build_projection_matrices, save_calibration, load_calibration
from pose3d.body.smplerx_wrapper import SMPLerXWrapper
from pose3d.fuse.view_selector import fuse_body_frame
from pose3d.fuse.hand_attach import attach_hands
from pose3d.fuse.temporal import smooth_poses
from pose3d.schema import BODY_JOINT_NAMES, hand_joint_names
from pose3d.viz import skeleton_viewer as viz

VIEWS_DEFAULT = ["H", "L", "R"]


def load_config(path: str, recording: str | None) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if recording:
        cfg["recording"] = recording
    return cfg


def resolve_paths(cfg: dict):
    # DataCapture/ = two levels up from this file (PoseEstimation/pose3d/run_pipeline.py)
    here = os.path.dirname(os.path.abspath(__file__))
    dcap = os.path.abspath(os.path.join(here, "..", ".."))
    root = cfg["dataset_root"]
    root = root if os.path.isabs(root) else os.path.join(dcap, root)   # -> DataCapture/dataset
    corpus = cfg["corpus"]       # Ego4WholeBody
    rec = cfg["recording"]
    rec_dir = os.path.join(root, corpus, rec)
    cali_dir = os.path.join(root, corpus, cfg["calibration_recording"])
    smplx = cfg["smplx_model_dir"]
    smplx_dir = smplx if os.path.isabs(smplx) else os.path.join(root, smplx)
    out_rel = cfg["output"]["dir"].format(recording=rec)
    out_dir = out_rel if os.path.isabs(out_rel) else os.path.join(root, out_rel)
    os.makedirs(out_dir, exist_ok=True)
    return rec_dir, cali_dir, smplx_dir, out_dir


def _align_script_abs(cfg):
    """Absolute path to Data_Preprocessing/align_audio.py, resolved from this
    file's location (DataCapture/PoseEstimation/pose3d/ -> ../../ = DataCapture/)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", cfg["align"]["script"]))


def step_align(cfg, rec_dir, views):
    a = cfg["align"]
    script = _align_script_abs(cfg)
    run = a.get("enabled", True)
    if not run:
        print("[pipeline] alignment disabled; using existing *_aligned.MP4 (if any).")
    return align_recording(rec_dir, views, cfg["reference"], cfg["video_ext"],
                           script, sr=a["sr"], run=run)


def step_calibrate(cfg, cali_dir, out_dir, views):
    cc = cfg["calibration"]
    calib_path = os.path.join(out_dir, cc.get("out", "calibration.json"))
    if not cc.get("enabled", True):
        raise RuntimeError("Calibration is required for metric 3D on the main line.")
    if os.path.isfile(calib_path):
        print(f"[pipeline] reusing calibration: {calib_path}")
        return load_calibration(calib_path)
    print("[pipeline] running multi-view calibration from", cali_dir)
    # Align cali videos too. They may lack an audio track (DESIGN warns) — if so,
    # audio alignment fails and we fall back to frame-index pairing (assumes the
    # cali videos were recorded synchronously / hardware-synced).
    ext = cfg["video_ext"]
    cali_paths = {cfg["reference"]: os.path.join(cali_dir, f"{cfg['reference']}{ext}")}
    for v in views:
        if v == cfg["reference"]:
            continue
        cali_paths[v] = os.path.join(cali_dir, f"{v}_aligned{ext}")
    try:
        align_recording(cali_dir, views, cfg["reference"], ext,
                        _align_script_abs(cfg), sr=cfg["align"]["sr"], run=True)
    except subprocess.CalledProcessError as e:
        print(f"[pipeline] cali audio alignment failed (no audio track? {e}). "
              f"Falling back to frame-index pairing — cali videos MUST be synced.")
        for v in views:
            if v != cfg["reference"]:
                cali_paths[v] = os.path.join(cali_dir, f"{v}{ext}")  # use originals
    calib = calibrate_multicam(cali_paths, cc)
    save_calibration(calib, calib_path)
    print(f"[pipeline] calibration saved: {calib_path}")
    print(f"           rms: {calib['rms']}")
    return calib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(os.path.dirname(__file__), "config.yaml"))
    ap.add_argument("--recording", default=None, help="recording dir name under <corpus>/")
    ap.add_argument("--device", default=None, help="cuda | cpu (overrides config)")
    ap.add_argument("--no-align", action="store_true", help="skip alignment (reuse *_aligned.MP4)")
    ap.add_argument("--no-calib", action="store_true", help="skip calibration (needs existing calibration.json)")
    ap.add_argument("--fusion", default="triangulate", choices=["triangulate", "global_fit"],
                    help="fusion backend; 'global_fit' is scaffold-only (raises NotImplementedError)")
    ap.add_argument("--max-frames", type=int, default=None, help="debug: cap frames processed")
    args = ap.parse_args()

    cfg = load_config(args.config, args.recording)
    if args.device:
        cfg["smplerx"]["device"] = args.device
    if args.no_align:
        cfg["align"]["enabled"] = False
    if args.no_calib:
        cfg["calibration"]["enabled"] = False

    if args.fusion == "global_fit":
        # SCAFFOLD: exercise the import, then fail fast with guidance. The default
        # triangulate path below is untouched. TODO: real fitting (see plan).
        from pose3d.fit import MultiViewFitter  # noqa: F401
        raise NotImplementedError(
            "--fusion global_fit is scaffold-only this round (fitting loop + losses "
            "are TODO, pending cali re-record + VPoser). Use --fusion triangulate "
            "(default). See docs/MULTIVIEW_FITTING_PLAN.md.")

    views = cfg.get("views", VIEWS_DEFAULT)
    rec_dir, cali_dir, smplx_dir, out_dir = resolve_paths(cfg)
    print(f"[pipeline] recording={cfg['recording']} out={out_dir}")

    # ---- 1. align ----
    align_info = step_align(cfg, rec_dir, views)
    print(f"[pipeline] delays(s)={align_info['delays']}")
    aligned = align_info["aligned_paths"]
    for v, p in aligned.items():
        assert os.path.isfile(p), f"aligned video missing: {p}"

    # ---- 2. calibrate ----
    calib = step_calibrate(cfg, cali_dir, out_dir, views)
    Ps = build_projection_matrices(calib)

    # ---- 3. SMPLer-X ----
    print("[pipeline] loading SMPLer-X (body+hands) ...")
    smplerx_cfg = dict(cfg["smplerx"])
    smplerx_cfg["repo_dir"] = os.path.join(os.path.dirname(__file__), smplerx_cfg["repo_dir"])
    smplerx_cfg["gender"] = cfg["smplx_gender"]
    wrapper = SMPLerXWrapper(smplerx_cfg, smplx_dir, device=smplerx_cfg["device"])

    use_mesh_tips = bool(cfg.get("hand", {}).get("use_mesh_tips", False))
    fuse_cfg = cfg

    # ---- 4. per-frame lockstep inference + fuse ----
    readers = {v: VideoReader(aligned[v]) for v in views}
    n = min((readers[v].n_frames or 10**9) for v in views)
    if args.max_frames:
        n = min(n, args.max_frames)
    fps = cfg.get("fps") or readers[cfg["reference"]].fps or 59.94
    gens = {v: readers[v].iter_frames(step=1) for v in views}

    frames_out = []
    # n may be unreliable on HEVC (CAP_PROP_FRAME_COUNT can be 0). Guard: if all
    # 3 generators are exhausted we stop, so a bad n can't create an empty-frame bomb.
    for i in tqdm.tqdm(range(n), desc="pose3d frames"):
        rgb_per_view = {}
        any_ok = False
        for v in views:
            try:
                _idx, rgb = next(gens[v])
                any_ok = True
            except StopIteration:
                rgb = None
            rgb_per_view[v] = rgb
        if not any_ok:
            print(f"[pipeline] all views exhausted at frame {i}; stopping.")
            break
        per_view = {}
        for v in views:
            rgb = rgb_per_view[v]
            if rgb is None:
                continue
            rec = wrapper.infer_frame(rgb)
            rec["view"] = v
            rec["_rgb_path"] = aligned[v]
            if not use_mesh_tips:
                rec.pop("verts_smplx", None)
            per_view[v] = rec

        fused_body = fuse_body_frame(per_view, Ps, calib, fuse_cfg)
        hands = attach_hands(per_view, fused_body, calib, use_mesh_tips=use_mesh_tips)
        primary = max(per_view, key=lambda v: per_view[v].get("det_score", 0.0)) if per_view else cfg["reference"]
        frames_out.append(make_frame(i / fps, i, primary, fused_body, hands))

    # ---- 5. temporal smoothing ----
    ts_cfg = cfg.get("fusion", {}).get("temporal_smoothing", {})
    if ts_cfg.get("enabled", True) and len(frames_out) >= 4:
        all_joint_names = (BODY_JOINT_NAMES
                           + hand_joint_names("L")
                           + hand_joint_names("R"))
        cutoff = float(ts_cfg.get("cutoff_hz", 6.0))
        order = int(ts_cfg.get("order", 3))
        src_win = int(ts_cfg.get("source_window", 5))
        print(f"[pipeline] temporal smoothing: cutoff={cutoff}Hz order={order} "
              f"window={src_win}")
        # Extract joint dicts from make_frame output for smoothing
        raw_joints = [f["joints"] for f in frames_out]
        smoothed = smooth_poses(raw_joints, all_joint_names,
                                fps=fps, cutoff_hz=cutoff, order=order,
                                source_window=src_win)
        for f, sj in zip(frames_out, smoothed):
            f["joints"] = sj

    # ---- 6. write poses.json ----
    poses_path = os.path.join(out_dir, cfg["output"]["poses_json"])
    header = {"version": "1.0", "fps": fps, "timeline_master": cfg["reference"],
              "scale": "metric_meters",
              "board_square_size_m": cfg["calibration"]["board_square_size_m"]}
    write_poses(poses_path, frames_out, header)
    print(f"[pipeline] wrote {poses_path}  ({len(frames_out)} frames)")
    _report_coverage(frames_out)

    # ---- 7. viz ----
    viz_dir = os.path.join(out_dir, cfg["output"].get("viz_dir", "viz"))
    nv = cfg["output"].get("viz_num_frames", 12)
    step = max(1, len(frames_out) // nv)
    sample_idx = list(range(0, len(frames_out), step))[:nv]
    # 2D overlays need the projected joints; re-infer a few frames (cheap) for viz
    print(f"[pipeline] writing viz to {viz_dir}")
    for v in views:
        # re-open to read sampled frames
        with VideoReader(aligned[v]) as r:
            for idx in sample_idx:
                rgb = r.read(idx)
                if rgb is None:
                    continue
                rec = wrapper.infer_frame(rgb)
                if rec.get("has_person"):
                    os.makedirs(viz_dir, exist_ok=True)
                    vis = viz.overlay_body2d(rgb, rec["body_joints2d"], rec.get("det_score", 0.0))
                    cv2.imwrite(os.path.join(viz_dir, f"{v}_{idx:06d}.png"),
                                cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    # 3D fused body for a couple of frames
    for idx in sample_idx[:3]:
        f = frames_out[idx]
        j3d = {k: (v["xyz"] if isinstance(v, dict) else None) for k, v in f["joints"].items()}
        viz.plot_body3d(j3d, os.path.join(viz_dir, f"body3d_{idx:06d}.png"))
    print("[pipeline] done.")


def _report_coverage(frames):
    names = BODY_JOINT_NAMES + hand_joint_names("L") + hand_joint_names("R")
    src_counts = Counter()
    for f in frames:
        for jn in names:
            s = f["joints"].get(jn, {}).get("source", "missing")
            src_counts[s] += 1
    total = len(frames) * len(names) or 1
    print("[coverage] per-source joint rate:")
    for s, c in sorted(src_counts.items(), key=lambda x: -x[1]):
        print(f"    {s:14s} {c/total*100:5.1f}%  ({c})")


if __name__ == "__main__":
    main()
