"""
pose3d.hand.smplx_hand_mapping — SMPL-X hand joints (16) -> our 27 names/side.

Inputs per side (from smplerx_wrapper.infer_frame):
  * joints3d_smplx: full (J,3) — wrist at 20/21, hand joints 22..36 (L) / 37..51 (R)
  * verts_smplx:    full (V,3) mesh (for optional mesh-based fingertips)

Mapping (per DESIGN.md §4.1, schema locked at 27/hand = 78 total):
  Wrist            <- smplx wrist joint                 (video)
  {finger}_MCP/CMC <- smplx hand joint 1 of finger      (video)
  {finger}_PIP     <- smplx hand joint 2 of finger      (video)  [thumb: Thumb_MCP]
  {finger}_DIP     <- smplx hand joint 3 of finger      (video)  [thumb: Thumb_IP]
  {finger}_Tip     <- mesh fingertip vertex OR extrapolation (video/derived)
  {finger}_Extra   <- midpoint of PIP-DIP phalanx        (derived)
  Palm_Center      <- centroid of Wrist + 4 MCPs         (derived)

TIPS: SMPL-X does NOT expose fingertip joints, so by default tips are
EXTRAPOLATED along the last phalanx (source=derived). Set
config.hand.use_mesh_tips=True AND verify FINGERTIP_VERTS for your SMPL-X
version to get source=video tips from the mesh.
"""
from __future__ import annotations
import numpy as np

from ..schema import HAND_LAYOUT, HAND_JOINTS_BASE, FINGER_SEGMENTS, PALM_MCP_NAMES, Source

# SMPL-X fingertip vertex indices. VERIFY against your SMPL-X .npz version
# before enabling config.hand.use_mesh_tips. Left indices mirrored.
FINGERTIP_VERTS = {
    "L": {"Thumb": 2746, "Index": 3039, "Middle": 2512, "Ring": 2047, "Pinky": 1599},
    "R": {"Thumb": 6514, "Index": 6807, "Middle": 6280, "Ring": 5815, "Pinky": 5367},
}
TIP_EXTRAP_RATIO = 0.6   # how far past the distal joint the tip sits (of last phalanx)
EXTRA_RATIO = 0.5        # *_Extra position along the PIP-DIP phalanx


def _interp(a, b, t):
    return (1 - t) * np.asarray(a, float) + t * np.asarray(b, float)


def map_hand(side: str, joints3d_smplx: np.ndarray, verts_smplx: np.ndarray | None,
             use_mesh_tips: bool = False) -> dict:
    """Return {prefixed_name: {"xyz":(3,), "conf":f, "source":str}} for one side.

    `side` in {'L','R'}. conf for smplx-derived joints is the view det_score
    proxy (set by caller via a separate pass); here we leave conf=1.0 for video
    joints and let the caller stamp the real per-view confidence.
    """
    side = side.upper()
    layout = HAND_LAYOUT[side]
    out: dict[str, dict] = {}

    # 1. direct joints (wrist + 15 phalanges) -> video
    for name, (kind, idx) in layout.items():
        if kind in ("wrist", "joint") and idx is not None:
            out[name] = {"xyz": np.asarray(joints3d_smplx[idx], float).copy(),
                         "conf": 1.0, "source": Source.VIDEO.value}

    # 2. fingertips (mesh vertex -> video, else extrapolation -> derived)
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        tip_name = f"{finger}_Tip"
        tip = None; src = Source.DERIVED.value
        if use_mesh_tips and verts_smplx is not None:
            vi = FINGERTIP_VERTS[side].get(finger)
            if vi is not None and vi < verts_smplx.shape[0]:
                v = verts_smplx[vi]
                if np.all(np.isfinite(v)):
                    tip, src = np.asarray(v, float).copy(), Source.VIDEO.value
        if tip is None:
            # extrapolate along the distal phalanx (proximal->distal->tip)
            seg = FINGER_SEGMENTS[finger]            # (mcp/cmc, pip/mcp, dip/ip)
            prox = out[seg[1]]["xyz"]; dist = out[seg[2]]["xyz"]
            tip = dist + (dist - prox) * TIP_EXTRAP_RATIO
        out[tip_name] = {"xyz": tip, "conf": 1.0, "source": src}

    # 3. *_Extra = midpoint of the PIP-DIP phalanx (thumb: MCP-IP) -> derived
    for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
        seg = FINGER_SEGMENTS[finger]
        a = out[seg[1]]["xyz"]; b = out[seg[2]]["xyz"]
        out[f"{finger}_Extra"] = {"xyz": _interp(a, b, EXTRA_RATIO),
                                   "conf": 1.0, "source": Source.DERIVED.value}

    # 4. Palm_Center = centroid of Wrist + 4 MCPs -> derived
    pts = [out["Wrist"]["xyz"]] + [out[m]["xyz"] for m in PALM_MCP_NAMES]
    out["Palm_Center"] = {"xyz": np.mean(np.stack(pts, 0), 0),
                          "conf": 1.0, "source": Source.DERIVED.value}

    # sanity: all 27 present
    assert set(out.keys()) == set(HAND_JOINTS_BASE), \
        f"hand mapping mismatch for {side}: missing {set(HAND_JOINTS_BASE)-set(out.keys())}"
    return out


def stamp_confidence(hand: dict, conf: float):
    """Stamp the per-view detection confidence onto video/derived joints."""
    for j in hand.values():
        j["conf"] = float(conf)
    return hand
