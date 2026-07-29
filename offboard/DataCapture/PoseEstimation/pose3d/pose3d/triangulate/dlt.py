"""
pose3d.triangulate.dlt — cross-view DLT triangulation -> metric 3D.

Confidence-weighted DLT with iterative outlier rejection. Projection matrices
are constant across time (rigid rig).
"""
from __future__ import annotations
import numpy as np


def dlt(points2d: list[np.ndarray], Ps: list[np.ndarray],
        confs: list[float] | None = None) -> np.ndarray:
    """Confidence-weighted linear DLT. Returns X (3,) in metric H frame."""
    assert len(points2d) == len(Ps) and len(points2d) >= 2
    if confs is None:
        confs = [1.0] * len(points2d)
    A = []
    for (u, v), P, c in zip(points2d, Ps, confs):
        w = float(c) ** 2
        A.append(w * (u * P[2] - P[0]))
        A.append(w * (v * P[2] - P[1]))
    A = np.asarray(A, dtype=np.float64)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    if abs(X[3]) < 1e-12:
        return np.array([np.nan, np.nan, np.nan])
    X = X[:3] / X[3]
    return X


def reproj_errors(X: np.ndarray, points2d, Ps) -> np.ndarray:
    """Per-view pixel reprojection error for the 3D point X."""
    Xh = np.array([X[0], X[1], X[2], 1.0])
    errs = []
    for (u, v), P in zip(points2d, Ps):
        p = P @ Xh
        p = p[:2] / (p[2] + 1e-12)
        errs.append(float(np.hypot(p[0] - u, p[1] - v)))
    return np.asarray(errs)


def triangulate_joint(view_data: dict, Ps: dict,
                      conf_thr: float = 0.3, reproj_thr_px: float = 30.0,
                      min_views: int = 2):
    """Triangulate one joint across views with iterative outlier rejection.

    view_data: { view_name: (xy(2,), conf) }   (xy in pixels)
    Ps:        { view_name: 3x4 ndarray }
    Returns dict(X=(3,) meters|None, used_views, reproj_errs, mean_conf, status).
    """
    cand = [(v, xy, c) for v, (xy, c) in view_data.items()
            if c >= conf_thr and np.all(np.isfinite(xy))]
    if len(cand) < min_views:
        return {"X": None, "used_views": [], "reproj_errs": [],
                "mean_conf": float(np.mean([c for _, _, c in cand]) if cand else 0.0),
                "status": "too_few"}

    views = [v for v, _, _ in cand]
    pts = [xy for _, xy, _ in cand]
    Pm = [Ps[v] for v in views]
    weights = [c for _, _, c in cand]

    X = dlt(pts, Pm, weights)
    if not np.all(np.isfinite(X)):
        return {"X": None, "used_views": [], "reproj_errs": [],
                "mean_conf": float(np.mean([c for _, _, c in cand])),
                "status": "missing"}

    # Iterative outlier rejection with MAD-based robust threshold
    while len(views) > min_views:
        errs = reproj_errors(X, pts, Pm)
        max_err = float(errs.max())
        if max_err <= reproj_thr_px:
            break

        # If median error already exceeds the user threshold, the DLT
        # is pathological — force-drop the worst view regardless of MAD.
        median_err = float(np.median(errs))
        if median_err > reproj_thr_px:
            pass  # will drop below
        else:
            # Compute robust threshold from error distribution
            mad = float(np.median(np.abs(errs - median_err)))
            # 1.4826 scales MAD to match std for Gaussian; 3x is ~99.7%
            robust_thr = median_err + 3.0 * mad * 1.4826 if mad > 1e-6 else reproj_thr_px
            effective_thr = max(reproj_thr_px, robust_thr)
            if max_err <= effective_thr:
                break

        # Drop the worst view and re-solve
        worst = int(np.argmax(errs))
        keep = [i for i in range(len(views)) if i != worst]
        if len(keep) < min_views:
            break
        views = [views[i] for i in keep]
        pts = [pts[i] for i in keep]
        Pm = [Pm[i] for i in keep]
        weights = [weights[i] for i in keep]
        X = dlt(pts, Pm, weights)

        if not np.all(np.isfinite(X)):
            return {"X": None, "used_views": [], "reproj_errs": [],
                    "mean_conf": float(np.mean([c for v, _, c in cand if v in views])),
                    "status": "missing"}

    # Final finite check (degenerate geometry can yield NaN)
    if not np.all(np.isfinite(X)):
        return {"X": None, "used_views": [], "reproj_errs": [],
                "mean_conf": float(np.mean([c for v, _, c in cand if v in views])),
                "status": "missing"}

    final_errs = reproj_errors(X, pts, Pm)
    return {"X": X, "used_views": views,
            "reproj_errs": final_errs.tolist(),
            "mean_conf": float(np.mean([c for v, _, c in cand if v in views])),
            "status": "triangulated"}
