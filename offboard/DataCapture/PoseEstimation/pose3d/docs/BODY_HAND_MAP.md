# Body & Hand Joint Mapping Reference

Canonical source: `pose3d/schema.py`. Output total = **78** (24 body + 27/hand ×2).

## Body — 24 SMPL joints ← SMPLer-X (SMPL-X) indices

SMPLer-X regresses SMPL-X; `smplx` forward yields `out.joints` (B, 127, 3).
Indices 0..21 are body; `Left_Hand`/`Right_Hand` use the hand-segment landmark
(first hand joint, 22/37) — the palm-root anchor where hands attach (distinct
from `Left_Wrist`/`Right_Wrist` at 20/21).

| Name | SMPL-X idx |  | Name | SMPL-X idx |
|---|---|---|---|---|
| Pelvis | 0 |  | Neck | 12 |
| Left_Hip | 1 |  | Left_Collar | 13 |
| Right_Hip | 2 |  | Right_Collar | 14 |
| Spine1 | 3 |  | Head | 15 |
| Left_Knee | 4 |  | Left_Shoulder | 16 |
| Right_Knee | 5 |  | Right_Shoulder | 17 |
| Spine2 | 6 |  | Left_Elbow | 18 |
| Left_Ankle | 7 |  | Right_Elbow | 19 |
| Right_Ankle | 8 |  | Left_Wrist | 20 |
| Spine3 | 9 |  | Right_Wrist | 21 |
| Left_Foot | 10 |  | Left_Hand | 22 |
| Right_Foot | 11 |  | Right_Hand | 37 |

## Hand — 27 names/hand ← SMPL-X hand joints (15) + wrist + derived

SMPL-X hand joints per side (15, MANO order: index, middle, pinky, ring, thumb),
indices 22..36 (left) / 37..51 (right). Output names carry `L_`/`R_` prefix.

| Output name | kind | SMPL-X source | source field |
|---|---|---|---|
| `{L,R}_Wrist` | wrist | body wrist (20/21) | `video` |
| `_Thumb_CMC/MCP/IP` | joint | thumb 1/2/3 | `video` |
| `_Thumb_Tip` | tip | mesh vertex OR extrapolation | `video` (mesh) / `derived` |
| `_Thumb_Extra` | derived | midpoint(MCP–IP) | `derived` |
| `_{Index,Middle,Ring,Pinky}_MCP` | joint | finger 1 | `video` |
| `_{...}_PIP` | joint | finger 2 | `video` |
| `_{...}_DIP` | joint | finger 3 | `video` |
| `_{...}_Tip` | tip | mesh vertex OR extrapolation | `video` (mesh) / `derived` |
| `_{...}_Extra` | derived | midpoint(PIP–DIP) | `derived` |
| `_Palm_Center` | derived | centroid(Wrist + 4 MCPs) | `derived` |

Per hand: **16 video** (wrist + 15 phalanges) + **5 tips** + **5 Extra** + **1
Palm_Center** = **27**. Default tips are extrapolated (`derived`); set
`config.yaml: hand.use_mesh_tips=true` AND verify `FINGERTIP_VERTS` in
`pose3d/hand/smplx_hand_mapping.py` for your SMPL-X version to get `video` tips
from the mesh.

## Optional HaMeR mapping (hand.backend=hamer)

HaMeR → MANO → 21 keypoints (incl. fingertips). Direct 1:1 to 21 names (source
`hamer`); `*_Extra` + `Palm_Center` derived. See `pose3d/hand/hamer_wrapper.py`.

## Source field semantics

| source | meaning |
|---|---|
| `triangulated` | cross-view DLT → metric 3D (body), H frame, meters |
| `singleview` | best single-view SMPLer-X, metric-scaled (approx) |
| `video` | directly from per-view model (SMPLer-X / HaMeR) |
| `derived` | interpolated/centroid from other joints |
| `missing` | not recovered this frame |
