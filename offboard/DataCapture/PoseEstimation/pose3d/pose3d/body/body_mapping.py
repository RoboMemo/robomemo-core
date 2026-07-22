"""
pose3d.body.body_mapping — SMPLer-X (SMPL-X) joints -> our 24 body names.

SMPLer-X regresses SMPL-X; the smplx forward yields `out.joints` (B, 127, 3)
in the SMPLer-X weak-perspective camera frame (root-centered, meters, BEFORE
the predicted transl). Indices 0..21 are the body; we take the 24 names from
schema.BODY_JOINTS (name -> smplx index). Left_Hand/Right_Hand use the hand-
segment landmark (indices 22 / 37), the palm-root anchor where hands attach.
"""
from __future__ import annotations
import numpy as np
from ..schema import BODY_JOINTS


def extract_body(joints: np.ndarray) -> dict:
    """joints: (J,3) smplx forward output (single person). -> {name: xyz(3,)}."""
    out = {}
    for name, idx in BODY_JOINTS.items():
        out[name] = np.asarray(joints[idx], dtype=np.float64).copy()
    return out


def extract_body_batch(joints: np.ndarray) -> list[dict]:
    """joints: (B,J,3). -> list of per-frame dicts."""
    return [extract_body(j) for j in joints]
