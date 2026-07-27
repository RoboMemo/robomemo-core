"""
pose3d.viz.skeleton_viewer — self-check + fused-output visualization.

PRIMARY USE: overlay the SMPLer-X projected body joints (body_joints2d) on the
original frame to VERIFY the weak-perspective -> pixel projection convention in
smplerx_wrapper._project_to_pixels. If the projected joints sit on the person,
the convention is right and cross-view DLT triangulation will be consistent.
If they are systematically shifted/scaled, adjust _project_to_pixels.

Also provides a 3D skeleton plot (matplotlib) for inspecting the fused metric
body in the H frame.

This module also draws HANDS (27 joints x2, attached at the wrist) and supports
per-joint SOURCE color coding (triangulated / singleview / video / derived), so
the fused output in poses.json — including the 52 hand joints — is visible.
"""
from __future__ import annotations
import numpy as np

# SMPL kinematic-tree edges for drawing (subset, body)
BODY_EDGES = [
    ("Pelvis", "Left_Hip"), ("Pelvis", "Right_Hip"), ("Pelvis", "Spine1"),
    ("Left_Hip", "Left_Knee"), ("Left_Knee", "Left_Ankle"), ("Left_Ankle", "Left_Foot"),
    ("Right_Hip", "Right_Knee"), ("Right_Knee", "Right_Ankle"), ("Right_Ankle", "Right_Foot"),
    ("Spine1", "Spine2"), ("Spine2", "Spine3"), ("Spine3", "Neck"), ("Neck", "Head"),
    ("Neck", "Left_Collar"), ("Left_Collar", "Left_Shoulder"),
    ("Left_Shoulder", "Left_Elbow"), ("Left_Elbow", "Left_Wrist"),
    ("Neck", "Right_Collar"), ("Right_Collar", "Right_Shoulder"),
    ("Right_Shoulder", "Right_Elbow"), ("Right_Elbow", "Right_Wrist"),
]

# --- Hand finger chains (per side). Each finger: Wrist -> phalanx1 -> 2 -> 3 -> Tip.
# Thumb uses CMC/MCP/IP; the others MCP/PIP/DIP. Matches schema.HAND_JOINTS_BASE. ---
_FINGERS = [("Thumb", "CMC", "MCP", "IP"), ("Index", "MCP", "PIP", "DIP"),
            ("Middle", "MCP", "PIP", "DIP"), ("Ring", "MCP", "PIP", "DIP"),
            ("Pinky", "MCP", "PIP", "DIP")]

def _hand_edges(prefix: str):
    e = []
    for name, p1, p2, p3 in _FINGERS:
        e += [(prefix + "Wrist", prefix + f"{name}_{p1}"),
              (prefix + f"{name}_{p1}", prefix + f"{name}_{p2}"),
              (prefix + f"{name}_{p2}", prefix + f"{name}_{p3}"),
              (prefix + f"{name}_{p3}", prefix + f"{name}_Tip")]
    return e

HAND_EDGES = _hand_edges("L_") + _hand_edges("R_")

# Per-joint source -> color. Values are RGB; convert to BGR for cv2.
SOURCE_COLORS_RGB = {
    "triangulated": (0.10, 0.70, 0.10),   # green — cross-view DLT (body)
    "singleview":   (0.95, 0.55, 0.05),   # orange — body single-view fallback
    "video":        (0.15, 0.45, 0.95),   # blue — hand joints direct from model
    "derived":      (0.55, 0.55, 0.55),   # gray — interpolated/extrapolated
    "hamer":        (0.55, 0.20, 0.85),   # purple — HaMeR hand (optional)
    "missing":      (0.90, 0.10, 0.10),   # red — not recovered
}
DEFAULT_RGB = (0.10, 0.70, 0.10)

def _src_rgb(src):
    return SOURCE_COLORS_RGB.get(str(src), DEFAULT_RGB)


def overlay_body2d(rgb: np.ndarray, body_joints2d: dict, det_score: float = 0.0,
                   hand_joints2d: dict | None = None, sources: dict | None = None) -> np.ndarray:
    """Draw projected body (+ optional hand) joints & edges on an RGB frame.

    hand_joints2d: optional {name: xy} for the 52 hand joints (drawn thinner,
        cyan if no per-joint source, else source-colored).
    sources: optional {name: source_str} to color each joint by its fusion source.
    Returns RGB uint8.
    """
    import cv2
    vis = rgb.copy()
    if vis.dtype != np.uint8:
        vis = (np.clip(vis, 0, 255)).astype(np.uint8)
    # body edges
    for a, b in BODY_EDGES:
        if a in body_joints2d and b in body_joints2d:
            pa = np.asarray(body_joints2d[a]).reshape(-1)
            pb = np.asarray(body_joints2d[b]).reshape(-1)
            if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                         (0, 255, 0), 3, cv2.LINE_AA)
    # hand edges (thinner, distinct)
    if hand_joints2d:
        for a, b in HAND_EDGES:
            if a in hand_joints2d and b in hand_joints2d:
                pa = np.asarray(hand_joints2d[a]).reshape(-1)
                pb = np.asarray(hand_joints2d[b]).reshape(-1)
                if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
                    cv2.line(vis, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                             (255, 200, 0), 2, cv2.LINE_AA)
    # body joints (source-colored if sources given)
    for name, xy in body_joints2d.items():
        p = np.asarray(xy).reshape(-1)
        if p.size >= 2 and np.all(np.isfinite(p[:2])):
            col = _src_rgb(sources.get(name)) if sources else (0, 0, 255)
            col_bgr = (int(col[2] * 255), int(col[1] * 255), int(col[0] * 255))
            cv2.circle(vis, (int(p[0]), int(p[1])), 5, col_bgr, -1, cv2.LINE_AA)
    # hand joints (smaller)
    if hand_joints2d:
        for name, xy in hand_joints2d.items():
            p = np.asarray(xy).reshape(-1)
            if p.size >= 2 and np.all(np.isfinite(p[:2])):
                cv2.circle(vis, (int(p[0]), int(p[1])), 3, (255, 200, 0), -1, cv2.LINE_AA)
    cv2.putText(vis, f"det={det_score:.2f}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)
    return vis


def plot_body3d(joints_xyz: dict, out_path: str | None = None,
                sources: dict | None = None, title: str | None = None):
    """3D plot of the fused metric body + hands (H frame).

    joints_xyz: {name: xyz(3,)|None} — should include the 24 body AND 52 hand names
        (as written to poses.json) to render hands.
    sources: optional {name: source_str} -> per-joint color (fusion breakdown).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig = plt.figure(figsize=(6, 8))
    ax = fig.add_subplot(111, projection="3d")
    plot_pts = []
    for n, xyz in joints_xyz.items():
        if xyz is None:
            continue
        xyz = np.asarray(xyz, float)
        col = _src_rgb(sources.get(n)) if sources else "r"
        ax.scatter(xyz[0], xyz[2], -xyz[1], c=[col], s=10)
        plot_pts.append((xyz[0], xyz[2], -xyz[1]))
    plot_pts = np.asarray(plot_pts) if plot_pts else np.zeros((1, 3))
    for setter, c in ((ax.set_xlim, 0), (ax.set_ylim, 1), (ax.set_zlim, 2)):
        lo, hi = plot_pts[:, c].min(), plot_pts[:, c].max()
        mid = (lo + hi) / 2
        rng = max(hi - lo, 0.1)
        setter(mid - rng / 2, mid + rng / 2)
    # body edges
    for a, b in BODY_EDGES:
        if joints_xyz.get(a) is not None and joints_xyz.get(b) is not None:
            pa, pb = np.asarray(joints_xyz[a]), np.asarray(joints_xyz[b])
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], [-pa[1], -pb[1]], "g-", lw=1.5)
    # hand edges (thinner, blue)
    for a, b in HAND_EDGES:
        if joints_xyz.get(a) is not None and joints_xyz.get(b) is not None:
            pa, pb = np.asarray(joints_xyz[a]), np.asarray(joints_xyz[b])
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], [-pa[1], -pb[1]], "-", color=(0.15, 0.45, 0.95), lw=0.8)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Z (m)"); ax.set_zlabel("-Y (m)")
    if title:
        ax.set_title(title, fontsize=10)
    if sources:
        legend = [Line2D([0], [0], marker="o", color="w", label="triangulated (body DLT)",
                         markerfacecolor=_src_rgb("triangulated"), markersize=8),
                  Line2D([0], [0], marker="o", color="w", label="singleview (body fallback)",
                         markerfacecolor=_src_rgb("singleview"), markersize=8),
                  Line2D([0], [0], marker="o", color="w", label="video (hand, model)",
                         markerfacecolor=_src_rgb("video"), markersize=8),
                  Line2D([0], [0], marker="o", color="w", label="derived",
                         markerfacecolor=_src_rgb("derived"), markersize=8)]
        ax.legend(handles=legend, loc="upper left", fontsize=7)
    plt.tight_layout()
    if out_path:
        plt.savefig(out_path, dpi=120)
    plt.close(fig)
