"""
pose3d.fuse.view_selector — per-frame body fusion (LOCKED policy A).

Body joints (24): DLT triangulate across views -> metric 3D in the H frame
(source=triangulated). When fewer than min_views see a joint (FOV non-overlap),
fall back to the best single view's SMPLer-X 3D scaled to metric via body-shape
beta and rigid-transformed into the H frame (source=singleview, approximate).
Hands are NOT triangulated (too small) — handled by hand_attach.py.

The relative extrinsics in `calib` are exactly the cam->H rigid transforms
(X_H = R_c X_cam + t_c), so single-view metric points land in the H frame.
"""
from __future__ import annotations
import numpy as np

from ..triangulate.dlt import triangulate_joint
from ..schema import BODY_JOINTS, BODY_JOINT_NAMES, Source

# Approx neutral SMPL body height (m). Used only for single-view fallback scale.
# height_from_beta applies a small correction from betas[0].
NEUTRAL_HEIGHT_M = 1.70
BETA0_HEIGHT_K = 0.06   # ~6 cm per unit of beta[0] (rough SMPL empirical)


def height_from_beta(betas) -> float:
    b0 = float(np.asarray(betas).reshape(-1)[0]) if betas is not None else 0.0
    return NEUTRAL_HEIGHT_M + BETA0_HEIGHT_K * b0


def similarity_align(A: np.ndarray, B: np.ndarray):
    """Similarity Procrustes: minimize || s*R@A + t - B ||. Returns (s, R, t)."""
    A = np.asarray(A, float); B = np.asarray(B, float)
    muA, muB = A.mean(0), B.mean(0)
    A0 = A - muA; B0 = B - muB
    H = A0.T @ B0
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    denom = float((A0 ** 2).sum())
    s = float(S.sum() / denom) if denom > 1e-9 else 1.0
    t = muB - s * (R @ muA)
    return s, R, t


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
        if len(out) >= 4:
            tri_anchors = {n: d["xyz"] for n, d in out.items()}

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
            A = np.stack([tri_anchors[n] for n in common])
            B = np.stack([sv[n]["xyz"] for n in common])
            res = float(np.linalg.norm(A - B) / max(np.linalg.norm(A), 1e-6))
            if res < best_res:
                best, best_res = v, res
        if best is not None:
            return best
    return max(present, key=lambda v: present[v].get("det_score", 0.0))


def _single_view_metric(rec, tri_anchors):
    """Scale SMPLer-X root-centered 3D to metric and place in the H frame.

    If a triangulated body exists, Procrustes-align this view's root-centered
    body to it -> exact metric H-frame placement (the alignment's R,t absorb the
    cam->H rotation+translation). Otherwise use body-shape beta height for scale
    and keep the root at the H-frame origin (absolute translation is not
    observable from a single view; proportions/scale are still metric).
    """
    units = np.asarray(rec["joints3d_smplx"], float)        # (J,3) root-centered
    A_all = np.stack([units[BODY_JOINTS[n]] for n in BODY_JOINT_NAMES])
    if tri_anchors is not None and len(tri_anchors) >= 4:
        common = [n for n in BODY_JOINT_NAMES if n in tri_anchors]
        A2 = np.stack([units[BODY_JOINTS[n]] for n in common])
        B = np.stack([tri_anchors[n] for n in common])
        s, Rr, tt = similarity_align(A2, B)
        H_pts = s * (A_all @ Rr.T) + tt
    else:
        s = height_from_beta(rec.get("betas")) / max(_body_height_units(units), 1e-6)
        H_pts = s * A_all          # root at H origin; metric proportions only
    conf = float(rec.get("det_score", 0.0))
    return {n: {"xyz": H_pts[i], "conf": conf}
            for i, n in enumerate(BODY_JOINT_NAMES)}
