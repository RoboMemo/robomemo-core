#!/usr/bin/env python3
"""Render the FUSED 3D pose (the actual pipeline output) from poses.json as a video.

Shows body (green edges) + hands (blue edges), every joint colored by its fusion
SOURCE (triangulated / singleview / video / derived), with a legend + frame info.
This is THE merged/fused pose (not 3 per-view predictions) — answers "is it fused?"
and "where are the hands?". Built purely from poses.json (no model, no re-inference).

Output: <pose3d_out>/fused_pose_3d.mp4
"""
import json, os, sys, glob
import numpy as np

PKG = os.path.expanduser(
    "~/robomemo-core/offboard/DataCapture/PoseEstimation/pose3d")
sys.path.insert(0, PKG)
from pose3d.schema import all_joint_names
from pose3d.viz.skeleton_viewer import (BODY_EDGES, HAND_EDGES,
                                        SOURCE_COLORS_RGB, DEFAULT_RGB)

OUT = os.path.expanduser(
    "~/robomemo-core/offboard/DataCapture/dataset/Ego4WholeBody/0721-1/pose3d_out")
POSES = os.path.join(OUT, "poses.json")
N_SAMPLES = 80
FPS = 6

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import cv2

d = json.load(open(POSES))
frames = d["frames"]
print(f"loaded {len(frames)} frames, n_joints={d['n_joints']}")

idxs = np.linspace(0, len(frames) - 1, N_SAMPLES).astype(int)

# global bounds in plot coords (x=X, y=Z, z=-Y) for a stable video
allpts = []
for fi in idxs:
    for jv in frames[fi]["joints"].values():
        if jv and jv.get("xyz"):
            x, y, z = jv["xyz"]
            allpts.append((x, z, -y))
allpts = np.asarray(allpts)

def rng(c):
    # robust bounds (2-98 pctile): degenerate triangulated joints from the poor
    # calibration can reach thousands of m (991 outliers >20m); min/max would
    # compress the real skeleton to a dot. Percentile bounds ignore those.
    lo, hi = float(np.percentile(allpts[:, c], 2)), float(np.percentile(allpts[:, c], 98))
    mid = (lo + hi) / 2
    r = max(hi - lo, 0.3)
    return mid - r / 2 - 0.15, mid + r / 2 + 0.15

xlim, ylim, zlim = rng(0), rng(1), rng(2)

TMP = "/tmp/fused_frames"
os.makedirs(TMP, exist_ok=True)
for k in TMP and glob.glob(os.path.join(TMP, "*.png")):
    os.remove(k)

def xyz_of(j, n, max_r=15.0):
    v = j.get(n)
    if not v or not v.get("xyz"):
        return None
    p = np.asarray(v["xyz"], float)
    # skip implausible joints (degenerate triangulation from poor calibration
    # can reach 100s-1000s of m) so they don't create stray points/edges.
    if not np.all(np.isfinite(p)) or np.max(np.abs(p)) > max_r:
        return None
    return p

for k, fi in enumerate(idxs):
    f = frames[fi]
    j = f["joints"]
    fig = plt.figure(figsize=(7.5, 8))
    ax = fig.add_subplot(111, projection="3d")
    for n in all_joint_names():
        p = xyz_of(j, n)
        if p is not None:
            src = (j.get(n) or {}).get("source")
            col = SOURCE_COLORS_RGB.get(str(src), DEFAULT_RGB)
            ax.scatter(p[0], p[2], -p[1], c=[col], s=12)
    for a, b in BODY_EDGES:
        pa, pb = xyz_of(j, a), xyz_of(j, b)
        if pa is not None and pb is not None:
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], [-pa[1], -pb[1]], "g-", lw=1.5)
    for a, b in HAND_EDGES:
        pa, pb = xyz_of(j, a), xyz_of(j, b)
        if pa is not None and pb is not None:
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], [-pa[1], -pb[1]], "-",
                    color=(0.15, 0.45, 0.95), lw=0.8)
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_zlim(*zlim)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("-Y (m)")
    ax.view_init(elev=12, azim=-60)
    ax.set_title(f"FUSED pose (H frame)  frame {f['frame_idx']}  t={f['t']:.2f}s  "
                 f"primary={f['primary_view']}", fontsize=9)
    legend = [
        Line2D([0], [0], marker="o", color="w", label="triangulated (body DLT)",
               markerfacecolor=SOURCE_COLORS_RGB["triangulated"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="singleview (body fallback)",
               markerfacecolor=SOURCE_COLORS_RGB["singleview"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="video (hand, model-direct)",
               markerfacecolor=SOURCE_COLORS_RGB["video"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="derived",
               markerfacecolor=SOURCE_COLORS_RGB["derived"], markersize=8),
    ]
    ax.legend(handles=legend, loc="upper left", fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(TMP, f"{k:05d}.png"), dpi=100)
    plt.close(fig)
print(f"rendered {len(idxs)} frames")

imgs = sorted(glob.glob(os.path.join(TMP, "*.png")))
im0 = cv2.imread(imgs[0]); h, w = im0.shape[:2]
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(os.path.join(OUT, "fused_pose_3d.mp4"), fourcc, FPS, (w, h))
for p in imgs:
    vw.write(cv2.imread(p))
vw.release()
sz = os.path.getsize(os.path.join(OUT, "fused_pose_3d.mp4"))
print(f"wrote fused_pose_3d.mp4  ({len(imgs)} frames, {w}x{h}, {sz} bytes)")
