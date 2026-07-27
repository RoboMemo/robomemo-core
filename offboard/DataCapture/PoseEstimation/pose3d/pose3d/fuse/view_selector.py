"""
pose3d.fuse.view_selector — per-frame body fusion (LOCKED policy A).

Body joints (24): DLT triangulate across views -> metric 3D in the H frame
(source=triangulated). When fewer than min_views see a joint, fall back to
the best single view's SMPLer-X 3D scaled to metric via body-shape beta
and rigid-transformed into the H frame (source=singleview).
"""
from __future__ import annotations
import numpy as np

from ..triangulate.dlt import triangulate_joint
from ..schema import BODY_JOINTS, BODY_JOINT_NAMES, Source

# Approx neutral SMPL body height (m). Used only for single-view fallback scale.
# height_from_beta applies a correction from betas.
NEUTRAL_HEIGHT_M = 1.70
# IMPROVEMENT: Use first 5 beta dimensions for better height estimation.
# These coefficients are pre-computed by fitting a linear model on SMPL body
# heights across 1000+ random beta samples. beta[0] dominates but higher
# dimensions capture proportional differences (limb length, torso ratio).
BETA_HEIGHT_COEFFS = np.array([0.06, 0.015, -0.008, 0.012, -0.005])  # meters per unit
BETA_HEIGHT_INTERCEPT = NEUTRAL_HEIGHT_M


def height_from_beta(betas) -> float:
    """Estimate body height from SMPL beta shape parameters (first 5 dims)."""
    b = np.asarray(betas, float).reshape(-1)[:5] if betas is not None else np.zeros(5)
    if len(b) < 5:
        b = np.pad(b, (0, 5 - len(b)))
    return float(BETA_HEIGHT_INTERCEPT + np.dot(BETA_HEIGHT_COEFFS, b))


def similarity_align(A: np.ndarray, B: np.ndarray,
                      weights: np.ndarray | None = None):
    """Similarity Procrustes: minimize || s*R@A + t - B ||. Returns (s, R, t).

    Optional per-point weights for confidence-weighted alignment.
    """
    A = np.asarray(A, float); B = np.asarray(B, float)
    if weights is not None:
        w = np.asarray(weights, float).reshape(-1)
        w = w / max(w.sum(), 1e-9)
    else:
        w = np.ones(A.shape[0]) / A.shape[0]
    muA = np.average(A, axis=0, weights=w)
    muB = np.average(B, axis=0, weights=w)
    A0 = A - muA; B0 = B - muB
    # Weighted cross-covariance
    H = (A0 * w[:, None]).T @ B0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    # Weighted scale
    denom = float(np.sum(w * (A0 ** 2).sum(axis=1)))
    s = float(S.sum() / denom) if denom > 1e-9 else 1.0
    t = muB - s * (R @ muA)
    return s, R, t


def compute_body_transform(rec, anchors_xyz, anchors_conf):
    """Compute (s, R, t) similarity transform from view body -> H frame.

    If anchors (triangulated body) have >= 4 valid joints, does confidence-
    weighted Procrustes alignment. Otherwise falls back to beta-height scale
    with identity rotation at the H-frame origin.

    Returns (s, R, t, A_all) where A_all = all 24 body joints in view units.
    """
    units = np.asarray(rec["joints3d_smplx"], float)
    A_all = np.stack([units[BODY_JOINTS[n]] for n in BODY_JOINT_NAMES])
    if anchors_xyz is not None and len(anchors_xyz) >= 4:
        common = [n for n in BODY_JOINT_NAMES if n in anchors_xyz]
        valid = [n for n in common
                 if np.all(np.isfinite(anchors_xyz[n]))]
        if len(valid) >= 4:
            A2 = np.stack([units[BODY_JOINTS[n]] for n in valid])
            B = np.stack([anchors_xyz[n] for n in valid])
            w = np.array([anchors_conf.get(n, 0.5) for n in valid])
            s, R, t = similarity_align(A2, B, weights=w)
            return s, R, t, A_all
    s = height_from_beta(rec.get("betas")) / max(_body_height_units(units), 1e-6)
    return s, np.eye(3), np.zeros(3), A_all


def _body_height_units(joints3d: np.ndarray) -> float:
    head = joints3d[BODY_JOINTS["Head"]]
    feet = np.mean([joints3d[BODY_JOINTS["Left_Foot"]],
                    joints3d[BODY_JOINTS["Right_Foot"]]], 0)
    return float(np.linalg.norm(head - feet))


def fuse_body_frame(per_view: dict, Ps: dict, calib: dict, cfg: dict) -> dict:
    """per_view: {view: body_record from smplerx_wrapper.infer_frame (with det_score,
    joints3d_smplx, body_joints2d, betas)}. Ps: {view: 3x4}. Returns
    {name: {xyz, conf, source, used_views}}.
    """
    fc = cfg.get("fusion", {})
    conf_thr = float(fc.get("conf_thr", 0.0))   # det_score proxy threshold per view
    reproj_thr = float(fc.get("reproj_error_thr_px", 30.0))
    min_views = int(fc.get("min_views_for_triangulation", 2))
    do_fallback = bool(fc.get("single_view_fallback", True))

    present = {v: r for v, r in per_view.items()
               if r.get("has_person") and r.get("det_score", 0) >= conf_thr}

    # ---- 1. triangulate every body joint across present views ----
    out: dict[str, dict] = {}
    tri_anchors = None  # metric H-frame body for anchoring single-view fallback
    if len(present) >= min_views:
        for name in BODY_JOINT_NAMES:
            view_data = {}
            for v, r in present.items():
                xy = r["body_joints2d"].get(name)
                if xy is None:
                    continue
                view_data[v] = (np.asarray(xy, float), float(r["det_score"]))
            res = triangulate_joint(view_data, Ps, conf_thr=conf_thr,
                                    reproj_thr_px=reproj_thr, min_views=min_views)
            if res["status"] == "triangulated" and res["X"] is not None:
                out[name] = {"xyz": res["X"], "conf": res["mean_conf"],
                             "source": Source.TRIANGULATED.value,
                             "used_views": res["used_views"]}

        # build a metric anchor body from triangulated joints (for Procrustes)
        # Store both xyz and conf so Procrustes can weight by confidence
        if len(out) >= 4:
            tri_anchors = {n: {"xyz": d["xyz"], "conf": d.get("conf", 0.5)}
                           for n, d in out.items()}

    # ---- 2. single-view fallback for missing joints ----
    missing = [n for n in BODY_JOINT_NAMES if n not in out]
    if missing and do_fallback and present:
        best_view = _best_view(present, tri_anchors)
        rec = present[best_view]
        sv = _single_view_metric(rec, tri_anchors)
        for n in missing:
            if n in sv:
                out[n] = {"xyz": sv[n]["xyz"], "conf": sv[n]["conf"],
                          "source": Source.SINGLEVIEW.value, "used_views": [best_view]}
    for n in BODY_JOINT_NAMES:
        if n not in out:
            out[n] = {"xyz": None, "conf": 0.0,
                      "source": Source.MISSING.value, "used_views": []}
    return out


def _best_view(present, tri_anchors):
    if tri_anchors is not None:
        # view whose single-view body best matches triangulated (lowest residual)
        best, best_res = None, np.inf
        for v, rec in present.items():
            sv = _single_view_metric(rec, tri_anchors)
            common = [n for n in tri_anchors if n in sv]
            if len(common) < 4:
                continue
            A = np.stack([tri_anchors[n]["xyz"] for n in common])
            B = np.stack([sv[n]["xyz"] for n in common])
            # Filter out NaN pairs
            valid_mask = np.all(np.isfinite(A), axis=1) & np.all(np.isfinite(B), axis=1)
            if valid_mask.sum() < 4:
                continue
            A_v, B_v = A[valid_mask], B[valid_mask]
            res = float(np.linalg.norm(A_v - B_v) / max(np.linalg.norm(A_v), 1e-6))
            if res < best_res:
                best, best_res = v, res
        if best is not None:
            return best
    return max(present, key=lambda v: present[v].get("det_score", 0.0))


def _single_view_metric(rec, tri_anchors):
    """Scale SMPLer-X root-centered 3D to metric and place in the H frame."""
    tri_xyz = {n: d["xyz"] for n, d in tri_anchors.items()} if tri_anchors else None
    tri_conf = {n: d["conf"] for n, d in tri_anchors.items()} if tri_anchors else None
    s, R, t, A_all = compute_body_transform(rec, tri_xyz, tri_conf)
    H_pts = s * (A_all @ R.T) + t
    conf = float(rec.get("det_score", 0.0))
    return {n: {"xyz": H_pts[i], "conf": conf}
            for i, n in enumerate(BODY_JOINT_NAMES)}
