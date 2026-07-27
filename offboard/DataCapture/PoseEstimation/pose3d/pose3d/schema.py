"""
pose3d.schema — joint schema, SMPL-X index mappings, source conventions.

Output per frame: 24 body + 27/hand x2 = 78 joints, each with 3D coords.
Body joints come from SMPLer-X -> SMPL-X forward; hands from SMPL-X hand
joints (SMPLer-X) optionally replaced by HaMeR/MANO.

NOTE: the user-locked hand schema has 27 names per hand (= 78 total), NOT 26.
We keep all 27 names verbatim per "精确名称，不要改".
"""

from __future__ import annotations
from collections import OrderedDict
from enum import Enum

# =====================================================================
# Source attribution (per-joint `source` field)
# =====================================================================
class Source(str, Enum):
    TRIANGULATED = "triangulated"  # cross-view DLT -> metric 3D (body)
    SINGLEVIEW   = "singleview"    # best single-view projection into H frame
    VIDEO        = "video"         # directly from per-view model (SMPLer-X/HaMeR)
    DERIVED      = "derived"       # interpolated/centroid from other joints
    PRIOR        = "prior"         # static prior (reserved, not used on main line)
    EXTERNAL     = "external"      # external provider (reserved)
    HAMER        = "hamer"         # optional HaMeR hand backend
    MISSING      = "missing"       # not recovered this frame

# =====================================================================
# Body: 24 SMPL-kinematic-tree joints  ->  SMPL-X joint indices
# SMPLer-X outputs SMPL-X; smplx forward yields joints where 0..21 are body,
# 22..36 left hand, 37..51 right hand. Left_Hand/Right_Hand map to the
# hand-segment landmark (first hand joint), the palm-root anchor where hands
# attach; they are distinct from Left_Wrist/Right_Wrist (20/21).
# =====================================================================
BODY_JOINTS: "OrderedDict[str, int]" = OrderedDict([
    ("Pelvis",       0),  ("Left_Hip",    1),  ("Right_Hip",   2),
    ("Spine1",       3),  ("Left_Knee",   4),  ("Right_Knee",  5),
    ("Spine2",       6),  ("Left_Ankle",  7),  ("Right_Ankle", 8),
    ("Spine3",       9),  ("Left_Foot",  10),  ("Right_Foot", 11),
    ("Neck",        12),  ("Left_Collar",13),  ("Right_Collar",14),
    ("Head",        15),  ("Left_Shoulder",16), ("Right_Shoulder",17),
    ("Left_Elbow",  18),  ("Right_Elbow",19),
    ("Left_Wrist",  20),  ("Right_Wrist",21),
    ("Left_Hand",   22),  ("Right_Hand", 37),
])
BODY_JOINT_NAMES = list(BODY_JOINTS.keys())
assert len(BODY_JOINT_NAMES) == 24

# =====================================================================
# Hand: 27 base names per hand (no L_/R_ prefix here; prefixed at output)
# Locked verbatim from the user spec. Order is anatomical (thumb..pinky),
# each finger = {base, mid, dist, tip, extra} (+ Wrist, Palm_Center).
# =====================================================================
HAND_JOINTS_BASE: list[str] = [
    "Wrist",
    # thumb (CMC/MCP/IP/Tip/Extra)
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip", "Thumb_Extra",
    # index
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip", "Index_Extra",
    # middle
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip", "Middle_Extra",
    # ring
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip", "Ring_Extra",
    # pinky
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip", "Pinky_Extra",
    # palm
    "Palm_Center",
]
assert len(HAND_JOINTS_BASE) == 27

LEFT_PREFIX, RIGHT_PREFIX = "L_", "R_"

def hand_joint_names(side: str) -> list[str]:
    """Return the 27 prefixed names for side in {'L','R'}."""
    p = LEFT_PREFIX if side.upper() == "L" else RIGHT_PREFIX
    return [p + n for n in HAND_JOINTS_BASE]

# =====================================================================
# SMPL-X hand joint index layout (per hand, 15 joints after wrist).
# smplx joints array: 22..36 = left hand, 37..51 = right hand. MANO order:
#   index(MCP,PIP,DIP), middle(...), pinky(...), ring(...), thumb(CMC,MCP,IP)
# `kind` tells the mapping module where each of our 27 names comes from.
#   - "joint" : directly from smplx joint array (source=video)
#   - "tip"   : fingertip, from SMPL-X mesh vertex or extrapolation
#   - "derived": interpolated/centroid (source=derived)
#   - "wrist" : the wrist landmark
# =====================================================================
def _hand_smplx_layout(side: str) -> "OrderedDict[str, tuple[str, int | None]]":
    base = 22 if side.upper() == "L" else 37   # first hand joint index
    wrist = 20 if side.upper() == "L" else 21
    # relative offsets within the 15 hand joints (index, middle, pinky, ring, thumb)
    idx_mcp, idx_pip, idx_dip = base + 0,  base + 1,  base + 2
    mid_mcp, mid_pip, mid_dip = base + 3,  base + 4,  base + 5
    pky_mcp, pky_pip, pky_dip = base + 6,  base + 7,  base + 8
    rng_mcp, rng_pip, rng_dip = base + 9,  base + 10, base + 11
    thb_cmc, thb_mcp, thb_ip  = base + 12, base + 13, base + 14
    return OrderedDict([
        ("Wrist",        ("wrist",  wrist)),
        ("Thumb_CMC",    ("joint",  thb_cmc)),
        ("Thumb_MCP",    ("joint",  thb_mcp)),
        ("Thumb_IP",     ("joint",  thb_ip)),
        ("Thumb_Tip",    ("tip",    None)),
        ("Thumb_Extra",  ("derived", None)),
        ("Index_MCP",    ("joint",  idx_mcp)),
        ("Index_PIP",    ("joint",  idx_pip)),
        ("Index_DIP",    ("joint",  idx_dip)),
        ("Index_Tip",    ("tip",    None)),
        ("Index_Extra",  ("derived", None)),
        ("Middle_MCP",   ("joint",  mid_mcp)),
        ("Middle_PIP",   ("joint",  mid_pip)),
        ("Middle_DIP",   ("joint",  mid_dip)),
        ("Middle_Tip",   ("tip",    None)),
        ("Middle_Extra", ("derived", None)),
        ("Ring_MCP",     ("joint",  rng_mcp)),
        ("Ring_PIP",     ("joint",  rng_pip)),
        ("Ring_DIP",     ("joint",  rng_dip)),
        ("Ring_Tip",     ("tip",    None)),
        ("Ring_Extra",   ("derived", None)),
        ("Pinky_MCP",    ("joint",  pky_mcp)),
        ("Pinky_PIP",    ("joint",  pky_pip)),
        ("Pinky_DIP",    ("joint",  pky_dip)),
        ("Pinky_Tip",    ("tip",    None)),
        ("Pinky_Extra",  ("derived", None)),
        ("Palm_Center",  ("derived", None)),
    ])

HAND_LAYOUT = {"L": _hand_smplx_layout("L"), "R": _hand_smplx_layout("R")}

# Fingers (base MCP/CMC, PIP, DIP/IP) used for per-finger tip extrapolation
# and *_Extra interpolation. Each entry: (proximal, distal) joint names whose
# segment defines the finger axis. *_Extra sits at the proximal phalanx.
FINGER_SEGMENTS = {
    "Thumb":  ("Thumb_CMC", "Thumb_MCP", "Thumb_IP"),       # 3 joints
    "Index":  ("Index_MCP", "Index_PIP", "Index_DIP"),
    "Middle": ("Middle_MCP", "Middle_PIP", "Middle_DIP"),
    "Ring":   ("Ring_MCP", "Ring_PIP", "Ring_DIP"),
    "Pinky":  ("Pinky_MCP", "Pinky_PIP", "Pinky_DIP"),
}

# MCP/CMC names whose centroid (with Wrist) defines Palm_Center
PALM_MCP_NAMES = ["Index_MCP", "Middle_MCP", "Ring_MCP", "Pinky_MCP"]

# =====================================================================
# Views & defaults
# =====================================================================
VIEW_NAMES = ["H", "L", "R"]
REFERENCE_VIEW = "H"
N_BODY = 24
N_HAND = 27
N_TOTAL = N_BODY + 2 * N_HAND  # 78

def all_joint_names() -> list[str]:
    """Full ordered list of 78 joint names as written to poses.json."""
    return BODY_JOINT_NAMES + hand_joint_names("L") + hand_joint_names("R")
