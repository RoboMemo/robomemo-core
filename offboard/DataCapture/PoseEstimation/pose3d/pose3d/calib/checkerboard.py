"""
pose3d.calib.checkerboard — checkerboard detection + metric object points.

Board: 11 inner-cols x 8 inner-rows, square = 20 mm (config board_square_size_m).

DETECTION uses findChessboardCornersSB first (Square-Based) — SB returns corners
in a CANONICAL order that is consistent across views/frames, which is what makes
stereo calibration solvable. We retry SB across flag+preprocessing combos so it
succeeds even on builds/data where the default call fails. Legacy
findChessboardCorners is the fallback; its 180-degree start-corner ambiguity is
canonicalized here, and any residual cross-view flip is finally resolved by the
stereo flip-test in multicam_calib.py.

OBJECT POINTS: empirically verified that findChessboardCorners(SB) returns
corners row-major over the (cols x rows) grid with index = row*cols + col, so
objectPoints[row*cols+col] must be (x=col, y=row)*sq. (The earlier
np.mgrid[0:rows,0:cols] form was TRANSPOSED vs the corner order and caused
catastrophic stereo RMS — see git history / CODE_REVIEW.)
"""
from __future__ import annotations
import cv2
import numpy as np

# termination criteria for sub-pixel refinement
_CRIT = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)


def object_points(pattern, square_size_m: float) -> np.ndarray:
    """3D board coordinates (Z=0 plane), in METERS, ordered to MATCH the corner
    index returned by findChessboardCorners(SB): k = row*cols + col -> (x=col,y=row).

    pattern = (cols, rows) of INNER corners (e.g. (11, 8)).
    """
    cw, ch = pattern
    grid = np.zeros((cw * ch, 3), dtype=np.float32)
    cols, rows = np.meshgrid(np.arange(cw), np.arange(ch))   # each shape (ch, cw)
    grid[:, 0] = cols.reshape(-1)   # x = col  (col varies fastest -> index row*cw+col)
    grid[:, 1] = rows.reshape(-1)   # y = row
    grid *= float(square_size_m)
    return grid


def _preprocs(gray: np.ndarray):
    """Yield (name, image) preprocessing variants for robust SB/legacy detection."""
    yield "raw", gray
    try:
        clahe = cv2.createCLAHE(3.0, (8, 8))
        yield "clahe", clahe.apply(gray)
    except Exception:
        pass
    yield "blur", cv2.GaussianBlur(gray, (5, 5), 0)
    yield "inv", 255 - gray


# SB flag combos to try (most permissive first); SB gives canonical orientation.
_SB_FLAGS = [
    cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE,
    cv2.CALIB_CB_NORMALIZE_IMAGE,
    0,
]


def detect(gray: np.ndarray, primary=(11, 8), fallback=(10, 7)):
    """Detect checkerboard inner corners in a grayscale uint8 image.

    Tries SB (canonical, cross-view-consistent order) across flag+preproc combos,
    then legacy findChessboardCorners (with deterministic canonicalization) as a
    fallback. Returns (corners_float32 Nx2, pattern_used) or (None, None).
    """
    for pat in (primary, fallback):
        # --- SB primary: canonical orientation ---
        sb = getattr(cv2, "findChessboardCornersSB", None)
        if sb is not None:
            for _pname, img in _preprocs(gray):
                for fl in _SB_FLAGS:
                    ok, corners = sb(img, pat, fl)
                    if ok and corners is not None and len(corners) == pat[0] * pat[1]:
                        corners = cv2.cornerSubPix(
                            gray, corners.astype(np.float32), (11, 11), (-1, -1), _CRIT)
                        return corners.reshape(-1, 2), pat
        # --- legacy fallback (deterministic orientation; residual flip resolved
        #     in multicam_calib's stereo flip-test) ---
        ok, corners = cv2.findChessboardCorners(
            gray, pat,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ok and corners is not None and len(corners) == pat[0] * pat[1]:
            corners = cv2.cornerSubPix(
                gray, corners.astype(np.float32), (11, 11), (-1, -1), _CRIT)
            return canonicalize(corners.reshape(-1, 2), pat), pat
    return None, None


def canonicalize(corners: np.ndarray, pattern) -> np.ndarray:
    """Deterministic orientation for legacy-detected corners.

    Resolves the 180-degree start-corner ambiguity by anchoring the grid so the
    first-row direction has a positive projection on the image-x axis. This is
    reproducible per view; the remaining cross-view 180 flip (if any) is resolved
    by the stereo flip-test in multicam_calib._resolve_stereo_flip().

    corners: (rows*cols, 2). pattern = (cols, rows).
    """
    cw, ch = pattern
    g = corners.reshape(ch, cw, 2).copy()
    # first-row direction (corner[0,0] -> corner[0,cols-1]); flip cols if it
    # points toward -x so the reading order goes +x.
    row_dir = g[0, -1] - g[0, 0]
    if row_dir[0] < 0:
        g = g[:, ::-1]
    # now decide row growth direction (g[0,0] -> g[rows-1,0]); flip rows so it
    # points toward +y (downward in image) -> deterministic start corner.
    col_dir = g[-1, 0] - g[0, 0]
    if col_dir[1] < 0:
        g = g[::-1, :]
    return g.reshape(-1, 2)


def draw(frame_bgr: np.ndarray, corners, pattern) -> np.ndarray:
    """Debug overlay."""
    vis = frame_bgr.copy()
    cv2.drawChessboardCorners(vis, pattern, corners, True)
    return vis
