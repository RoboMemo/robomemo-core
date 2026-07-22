"""
pose3d.triangulate.dlt — cross-view DLT triangulation -> metric 3D.

Coordinate system: the H camera frame, METERS (scale locked by the 20 mm
checkerboard). For each body joint we collect per-view 2D pixel detections
+ confidence, build the linear DLT system from the calibrated projection
matrices, and solve via SVD. Views whose reprojection error exceeds a
threshold are dropped as outliers and the solve is retried.

Projection matrices (constant across time — rigid rig):
    P_H = K_H [ I  | 0    ]
    P_L = K_L [ R_{HL} | t_{HL} ]
    P_R = K_R [ R_{HR} | t_{HR} ]
"""
from __future__ import annotations
import numpy as np


def projection_matrix(K, R, t):
    """3x4 projection P = K [R | t]. K:(3,3) R:(3,3) t:(3,) all numpy."""
    Rt = np.hstack([R, t.reshape(3, 1)])
    return K @ Rt


def dlt(points2d: list[np.ndarray], Ps: list[np.ndarray]) -> np.ndarray:
    """Linear DLT. points2d[i] = (u,v), Ps[i] = 3x4. Returns X (3,) in metric H frame."""
    assert len(points2d) == len(Ps) and len(points2d) >= 2
    A = []
    for (u, v), P in zip(points2d, Ps):
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A, dtype=np.float64)
    # Solve A X = 0 via SVD; X = last right singular vector
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
    """Triangulate one joint across views with outlier rejection.

    view_data: { view_name: (xy(2,), conf) }   (xy in pixels)
    Ps:        { view_name: 3x4 ndarray }
    Returns dict(X=(3,) meters|None, used_views, reproj_errs, mean_conf,
                 status) where status in {'triangulated','too_few','missing'}.
    """
    # keep views with sufficient confidence
    cand = [(v, xy, c) for v, (xy, c) in view_data.items()
            if c >= conf_thr and np.all(np.isfinite(xy))]
    if len(cand) < min_views:
        return {"X": None, "used_views": [], "reproj_errs": [],
                "mean_conf": float(np.mean([c for _, _, c in cand]) if cand else 0.0),
                "status": "too_few"}

    views = [v for v, _, _ in cand]
    pts = [xy for _, xy, _ in cand]
    Pm = [Ps[v] for v in views]

    X = dlt(pts, Pm)
    if not np.all(np.isfinite(X)):
        return {"X": None, "used_views": [], "reproj_errs": [],
                "mean_conf": float(np.mean([c for _, _, c in cand])),
                "status": "missing"}

    # outlier rejection: drop worst view if it blows the threshold, retry
    errs = reproj_errors(X, pts, Pm)
    if errs.max() > reproj_thr_px and len(views) > min_views:
        worst = int(np.argmax(errs))
        keep = [i for i in range(len(views)) if i != worst]
        if len(keep) >= min_views:
            views = [views[i] for i in keep]
            pts = [pts[i] for i in keep]
            Pm = [Pm[i] for i in keep]
            X = dlt(pts, Pm)
            errs = reproj_errors(X, pts, Pm)

    # recheck finite after retry (degenerate geometry can yield NaN)
    if not np.all(np.isfinite(X)):
        return {"X": None, "used_views": [], "reproj_errs": [],
                "mean_conf": float(np.mean([c for v, _, c in cand if v in views])),
                "status": "missing"}

    return {"X": X, "used_views": views,
            "reproj_errs": errs.tolist(),
            "mean_conf": float(np.mean([c for v, _, c in cand if v in views])),
            "status": "triangulated"}
