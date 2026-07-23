"""
pose3d.fuse.hand_attach — place single-view hands into the metric H frame.

Hands are too small to triangulate, so each hand keeps its SMPLer-X estimate
and is anchored onto the fused body via the body's similarity transform.
"""
from __future__ import annotations
import numpy as np

from ..schema import Source
from ..hand.smplx_hand_mapping import map_hand
from .view_selector import compute_body_transform


def attach_hands(per_view: dict, body_fused: dict, calib: dict,
                 use_mesh_tips: bool = False) -> dict:
    """Returns {side: {prefixed_name: {xyz, conf, source}}} for sides L, R."""
    sides = {"L": ("Left_Wrist", "Left_Hand"), "R": ("Right_Wrist", "Right_Hand")}
    result = {}

    # fused body anchors (valid = not None and finite) for Procrustes
    fused_xyz = {}
    fused_conf = {}
    for n, d in body_fused.items():
        if d.get("xyz") is not None and np.all(np.isfinite(d["xyz"])):
            fused_xyz[n] = d["xyz"]
            fused_conf[n] = float(d.get("conf", 0.5))

    for side, (wrist_name, hand_root_name) in sides.items():
        rec = _best_hand_view(per_view, side)
        if rec is None:
            result[side] = _empty_hand(side)
            continue

        # 1. body similarity (view root-centered -> H frame)
        s, R, t, A_all = compute_body_transform(rec, fused_xyz, fused_conf)

        # 2. hand joints in view frame -> metric H frame via the same transform
        units = np.asarray(rec["joints3d_smplx"], float)
        hand = map_hand(side, units, rec.get("verts_smplx"), use_mesh_tips=use_mesh_tips)
        for j in hand.values():
            j["xyz"] = s * (j["xyz"] @ R.T) + t

        # 3. snap wrist onto fused body wrist if available
        # IMPROVEMENT: Confidence-based blending. When the Procrustes alignment
        # is well-constrained (high body confidence), we trust the aligned wrist
        # position more. When body confidence is low, we blend more toward the
        # fused body wrist to avoid hand-body misalignment.
        target = body_fused.get(wrist_name, {}).get("xyz")
        target_conf = body_fused.get(wrist_name, {}).get("conf", 0.5)
        if target is not None:
            aligned_wrist = hand["Wrist"]["xyz"]
            # Blend factor: high body confidence -> more of the fused wrist
            # Low confidence -> more of the aligned wrist (from Procrustes)
            alpha = np.clip(target_conf, 0.2, 0.8)  # don't fully trust either
            blended_wrist = (1.0 - alpha) * aligned_wrist + alpha * np.asarray(target, float)
            shift = blended_wrist - aligned_wrist
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
    """Pick the best view for a specific hand side, with same-side camera preference."""
    # Side preference order: primary (same side), center (H), opposite
    side_preference = {"L": ["L", "H", "R"], "R": ["R", "H", "L"]}
    ranked = side_preference.get(side, ["H", "L", "R"])

    best, best_score = None, -1.0
    for v, rec in per_view.items():
        if not rec.get("has_person"):
            continue
        score = float(rec.get("det_score", 0.0))
        # Small bonus for the camera on the same side as the hand.
        # Magnitude 0.1 means a same-side cam wins over an opposite cam
        # only when detection scores are within ~0.1 of each other.
        if v in ranked:
            score += 0.1 * (3 - ranked.index(v))
        if score > best_score:
            best, best_score = rec, score
    return best


def _empty_hand(side):
    from ..schema import hand_joint_names
    return {n: {"xyz": None, "conf": 0.0, "source": Source.MISSING.value}
            for n in hand_joint_names(side)}
