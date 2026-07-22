"""
pose3d.fuse.hand_attach — place single-view hands into the metric H frame.

Hands are too small to triangulate from 3 head-cam views, so each hand keeps
its SMPLer-X single-view estimate (relative geometry), and is anchored onto the
(triangulated) body:
  1. Pick the best view for the hand (hand present + best det_score).
  2. Procrustes-align that view's root-centered BODY to the fused H-frame body
     -> similarity (s, R, t) that already encodes metric scale + cam->H transform.
     (When no triangulated body exists that frame, scale falls back to body-shape
     beta and the root sits at the H-frame origin.)
  3. Apply the SAME (s, R, t) to the view's hand joints (same smplx forward).
  4. Snap the hand Wrist onto the fused body wrist (Left_Wrist / Right_Wrist).

Body and hand come from one smplx forward per view, so they share units/frame
and the body similarity transform is valid for the hands.
"""
from __future__ import annotations
import numpy as np

from ..schema import BODY_JOINTS, BODY_JOINT_NAMES, Source
from ..hand.smplx_hand_mapping import map_hand
from .view_selector import similarity_align, height_from_beta, _body_height_units


def attach_hands(per_view: dict, body_fused: dict, calib: dict,
                 use_mesh_tips: bool = False) -> dict:
    """Returns {side: {prefixed_name: {xyz, conf, source}}} for sides L, R."""
    sides = {"L": ("Left_Wrist", "Left_Hand"), "R": ("Right_Wrist", "Right_Hand")}
    result = {}

    # fused body anchors (valid = not None and finite) for Procrustes
    fused = {n: d["xyz"] for n, d in body_fused.items()
             if d.get("xyz") is not None and np.all(np.isfinite(d["xyz"]))}
    fused_valid = len(fused) >= 4

    for side, (wrist_name, hand_root_name) in sides.items():
        rec = _best_hand_view(per_view, side)
        if rec is None:
            result[side] = _empty_hand(side)
            continue

        # 1. body similarity (view root-centered -> H frame)
        units = np.asarray(rec["joints3d_smplx"], float)
        A_all = np.stack([units[BODY_JOINTS[n]] for n in BODY_JOINT_NAMES])
        if fused_valid:
            common = [n for n in BODY_JOINT_NAMES if n in fused]
            A2 = np.stack([units[BODY_JOINTS[n]] for n in common])
            B = np.stack([fused[n] for n in common])
            s, R, t = similarity_align(A2, B)
        else:
            s = height_from_beta(rec.get("betas")) / max(_body_height_units(units), 1e-6)
            R, t = np.eye(3), np.zeros(3)

        # 2. hand joints in view frame -> metric H frame via the same transform
        hand = map_hand(side, units, rec.get("verts_smplx"), use_mesh_tips=use_mesh_tips)
        for j in hand.values():
            j["xyz"] = s * (j["xyz"] @ R.T) + t

        # 3. snap wrist onto fused body wrist if available
        target = body_fused.get(wrist_name, {}).get("xyz")
        if target is not None:
            shift = np.asarray(target, float) - hand["Wrist"]["xyz"]
            for j in hand.values():
                j["xyz"] = j["xyz"] + shift

        # stamp confidence from the view; keep per-joint source from mapping
        conf = float(rec.get("det_score", 0.0))
        for j in hand.values():
            j["conf"] = conf
        # emit with the side prefix (L_/R_) to match the output schema
        prefix = "L_" if side == "L" else "R_"
        result[side] = {prefix + name: d for name, d in hand.items()}
    return result


def _best_hand_view(per_view, side):
    # In this data the body+hands come together from SMPLer-X; pick the view
    # with a person and best score. (Per-hand visibility refinement is a TODO.)
    cands = [r for r in per_view.values() if r.get("has_person")]
    if not cands:
        return None
    return max(cands, key=lambda r: r.get("det_score", 0.0))


def _empty_hand(side):
    from ..schema import hand_joint_names
    return {n: {"xyz": None, "conf": 0.0, "source": Source.MISSING.value}
            for n in hand_joint_names(side)}
