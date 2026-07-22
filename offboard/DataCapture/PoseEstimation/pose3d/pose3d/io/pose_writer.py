"""
pose3d.io.pose_writer — serialize per-frame 3D poses to poses.json (exact schema).

Header: version, fps, timeline_master=H, scale=metric_meters, board_square_size_m,
n_joints, joint_names.
Each frame: t, frame_idx, primary_view, joints{ <name>: {xyz, conf, source} }.

78 joints = 24 body + 27/hand x2. xyz in METERS, H-camera reference frame.
"""
from __future__ import annotations
import json
import numpy as np

from ..schema import all_joint_names, N_TOTAL


def _xyz(v):
    if v is None:
        return None
    return [float(x) for x in np.asarray(v, float).reshape(-1)]


def write_poses(path: str, frames: list, header: dict, indent: int = 1):
    """frames: list of {t, frame_idx, primary_view, joints: {name: {xyz,conf,source}}}."""
    out = {
        "version": header.get("version", "1.0"),
        "fps": header.get("fps"),
        "timeline_master": header.get("timeline_master", "H"),
        "scale": header.get("scale", "metric_meters"),
        "board_square_size_m": header.get("board_square_size_m"),
        "n_joints": N_TOTAL,
        "joint_names": all_joint_names(),
        "source_legend": {
            "triangulated": "cross-view DLT -> metric 3D (body)",
            "singleview": "best single-view SMPLer-X, metric-scaled (approx)",
            "video": "directly from per-view model",
            "derived": "interpolated/centroid from other joints",
            "hamer": "optional HaMeR hand backend",
            "missing": "not recovered this frame",
        },
        "frames": frames,
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=indent)
    return path


def make_frame(t: float, frame_idx: int, primary_view: str,
               body_joints: dict, hand_joints: dict) -> dict:
    """Assemble one frame record.

    body_joints: {name: {xyz, conf, source, used_views?}}
    hand_joints: {side: {prefixed_name: {xyz, conf, source}}}
    """
    joints = {}
    for name, d in body_joints.items():
        joints[name] = {"xyz": _xyz(d.get("xyz")),
                        "conf": float(d.get("conf", 0.0)),
                        "source": d.get("source", "missing")}
    for side in ("L", "R"):
        for name, d in (hand_joints.get(side) or {}).items():
            joints[name] = {"xyz": _xyz(d.get("xyz")),
                            "conf": float(d.get("conf", 0.0)),
                            "source": d.get("source", "missing")}
    return {"t": float(t), "frame_idx": int(frame_idx),
            "primary_view": primary_view, "joints": joints}
