# pose3d — 3D Whole-Body Pose Pipeline (SMPLer-X + Multi-View DLT)

From **3 synchronized head-cam videos** (`H/L/R.MP4`) of a person (whole body),
produce per-frame **78-joint 3D poses** in **metric meters** (H-camera frame):
**24 SMPL body joints + 27 hand joints ×2**.

Pipeline: **audio time-align → checkerboard multi-view calibration → per-view
SMPLer-X-H32 (body+hands) → cross-view DLT triangulation (body, metric) →
single-view fallback + hand attach → `poses.json`**.

Target platform: **Linux + NVIDIA CUDA**. The dev Mac has no CUDA and only writes
code; 3D inference must run on the CUDA box. (`--no-calib` smoke path can do the
CPU-only alignment step on a Mac.)

---

## 0. One-minute summary

```bash
# on Linux + CUDA (RTX 5080 / Blackwell OK):
conda env create -f environment.yml && conda activate pose3d
bash download_models.sh          # clones SMPLer-X, H32 ckpt, yolov8n.pt, transformer_utils, checks SMPL-X
python run_pipeline.py --recording 0721-1
# -> dataset/Ego4WholeBody/0721-1/pose3d_out/poses.json (+ viz/)
```

> ⚠️ The ONE thing to resolve on a Blackwell GPU before `import` works: the
> SMPLer-X regressor imports `mmcv.ops.roi_align`, and mmcv-full 1.7.1 has no
> cu126/torch2.7 wheel. See §9 for the source-build vs torchvision-shim choice.

---

## 1. Prerequisites

- NVIDIA GPU + driver supporting **CUDA 12.6+** (cu126 build; cu128/torch-nightly
  also fine for Blackwell sm_120).
- **conda** (miniconda/miniforge), **git**, **git-lfs**,
  **huggingface-cli** (`pip install huggingface_hub`; the H32 ckpt is public but
  large, may need `huggingface-cli login` with a free token).
- **ffmpeg/ffprobe** on PATH (the alignment step shells out to them).

## 2. Install (Linux + CUDA, Blackwell)

```bash
conda env create -f environment.yml   # torch 2.7 + cu126, ultralytics, librosa, mmcv-full...
conda activate pose3d
```

Then `download_models.sh` installs SMPLer-X's bundled mmpose fork
(`pip install -v -e third_party/SMPLer-X/main/transformer_utils`) — do NOT
pip-install a stock mmpose (it conflicts).

**mmcv-full on Blackwell** (the one wrinkle): SMPLer-X needs mmcv 1.x
(`mmcv.Config`, `mmcv.runner`, `mmcv.ops.roi_align`). There is no prebuilt
cu126/torch2.7 wheel, so pick one (see §9): source-build mmcv-full, or shim
`mmcv.ops.roi_align` → `torchvision.ops.roi_align` with mmcv-lite. Detection is
**ultralytics YOLO** (no mmdet) — that part is clean.

**numpy is pinned to 1.26.4.** Do not let it upgrade to 2.x.

## 3. Models

```bash
bash download_models.sh
```

This:
1. `git clone`s **SMPLer-X** into `third_party/SMPLer-X` (set `SMPLERX_COMMIT=<sha>`
   to pin a commit for reproducibility).
2. Downloads the **SMPLer-X-H32\*** (camera-fix) checkpoint from HuggingFace
   (`caizhongang/SMPLer-X`) → `third_party/SMPLer-X/pretrained_models/smpler_x_h32_correct.pth.tar`.
3. Fetches the **ultralytics YOLO** person detector → `yolov8n.pt` (auto-downloads
   on first run too; `config.yaml: smplerx.detector` can point to yolov8s/m).
4. `pip install -v -e third_party/SMPLer-X/main/transformer_utils` — SMPLer-X's
   **bundled mmpose fork** (do NOT pip-install a stock mmpose; it conflicts).
5. **Verifies** the SMPL-X body model is present (already in this repo at
   `dataset/models/smplx/` — `SMPLX_NEUTRAL.npz` etc.) and symlinks it into the
   layout SMPLer-X expects (`common/utils/human_model_files/smplx`). **No download.**
6. Prints instructions for the **OPTIONAL** HaMeR/MANO step (disabled).

> SMPL-X body model is **already provided** (`dataset/models/smplx/`, both
> `.npz` and `.pkl`). SMPLer-X is pointed at it automatically.

> **SMPLer-X upstream stack.** The official repo is `python3.8 / torch1.12 /
> cu113`. This pipeline ships `torch2.0.1 / cu118` (for modern GPUs: RTX 30/40
> series need cu118+). If you hit mmpose `force=True` registration errors, follow
> the SMPLer-X README note (add `force=True` to the module registrations under
> `main/transformer_utils/mmpose/...`). On older GPUs you may alternatively use
> the official `cu113/torch1.12` wheels verbatim.

## 4. Data layout

```
dataset/
├── models/smplx/                 # SMPLX_NEUTRAL/MALE/FEMALE.npz  (provided)
└── Ego4WholeBody/
    ├── 0721-cali/                # CALIBRATION recording (checkerboard)
    │   ├── H.MP4  L.MP4  R.MP4
    └── 0721-1/                   # any recording you want to process
        ├── H.MP4  L.MP4  R.MP4   # 1920x1440 HEVC ~59.94fps, with audio
        └── pose3d_out/           # <- output lands here
```

Pipeline is generic: `python run_pipeline.py --recording <name>` for any
recording dir under `Ego4WholeBody/` containing `H/L/R.MP4`.

## 5. Calibration (the metric anchor)

- Board: **11 cols × 8 rows, square = 20 mm** (`config.yaml:
  calibration.board_square_size_m: 0.02`). This 20 mm is what locks the whole
  pipeline to **real meters** (DLT triangulation scale).
- Calibration recording: `Ego4WholeBody/0721-cali/{H,L,R}.MP4`.
- **OpenCV inner-corner gotcha**: `findChessboardCorners` takes the INNER-corner
  count. We try `(11,8)` first and fall back to `(10,7)` (see
  `pose3d/calib/checkerboard.py`). If neither detects, your board is different —
  update `config.yaml: calibration.board_cols/rows/pattern_fallback`.
- Calibration audio-aligns the 3 cali videos first (same as operating data), then
  per-camera `calibrateCamera` + pairwise `stereoCalibrate(H,L)/(H,R)` with
  `CALIB_FIX_INTRINSIC` → constant rigid relative extrinsics (meters).
- Result saved to `<out>/calibration.json` (reused on subsequent runs).

## 6. Run

```bash
python run_pipeline.py --recording 0721-1
# options:
#   --device cuda         (default; --device cpu only for tiny smoke, very slow)
#   --no-align            reuse existing *_aligned.MP4
#   --no-calib            skip calibration (needs existing calibration.json)
#   --max-frames N        debug: cap frames
```

What it does, per frame (lockstep across aligned H/L/R):
1. SMPLer-X-H32 → SMPL-X body + hand joints (per view).
2. Body: **DLT triangulate** across views → metric 3D (H frame). Non-overlap →
   beta-scaled single-view fallback (`source=singleview`, approximate).
3. Hands: single-view SMPLer-X estimate, **snapped onto the triangulated wrist**
   (`source=video/derived`). Hands are too small to triangulate from these cams.
4. Write `poses.json` + sample `viz/` overlays.

## 7. Output (`poses.json`)

```jsonc
{
  "version": "1.0", "fps": 59.94, "timeline_master": "H",
  "scale": "metric_meters", "board_square_size_m": 0.02,
  "n_joints": 78, "joint_names": [ "Pelvis", ..., "L_Wrist", ..., "R_Palm_Center" ],
  "source_legend": { "triangulated": "...", "singleview": "...", "derived": "...", ... },
  "frames": [
    { "t": 12.345, "frame_idx": 740, "primary_view": "H",
      "joints": {
        "Pelvis":      { "xyz": [0.12, -0.03, 2.41], "conf": 0.92, "source": "triangulated" },
        "Left_Wrist":  { "xyz": [-0.31, -0.20, 2.20], "conf": 0.84, "source": "triangulated" },
        "L_Wrist":     { "xyz": [-0.31, -0.20, 2.20], "conf": 0.90, "source": "video" },
        "L_Index_Tip": { "xyz": [-0.35, -0.24, 2.18], "conf": 0.60, "source": "derived" },
        "...": "..." } }
  ]
}
```

- 78 joints: **24 body** (names below) + **27/hand ×2** (`L_`/`R_` prefixed).
- `xyz` in **meters**, **H-camera reference frame** (per-frame; the rig moves
  with the head, so the frame is rigid at each instant).
- `source`: `triangulated` (body, metric), `singleview` (body fallback, approx),
  `video` (model-direct hand joint), `derived` (interpolated `*_Extra`,
  `Palm_Center`, and fingertip extrapolation by default), `missing`.

At the end the run prints per-source **coverage** (% of joint-slots).

## 8. ⚠️ VERIFY the SMPLer-X inference (do this first)

Two things that could not be tested on the dev Mac, and that you should confirm
on your **first frame** before trusting `poses.json`:

**(a) Demoer output keys.** The wrapper calls `demoer.model({'img':...}, {}, {},
'test')` and reads `smplx_root_pose / smplx_body_pose / smplx_lhand_pose /
smplx_rhand_pose / smplx_shape / cam_trans`. On your pinned commit, print
`out.keys()` once and confirm these exist (shapes: root (1,3), body (1,63),
hand (1,45), shape (1,10), cam_trans (1,3)). If a key differs, edit the adapter
method `_raw_inference()` in `pose3d/body/smplerx_wrapper.py` (marked [A2]).

**(b) Projection to pixels.** Joints are projected with the **perspective**
camera SMPLer-X derives per-person: `focal = cfg.focal/input_body_shape*bbox_wh`,
`princpt = cfg.princpt/input_body_shape*bbox_wh + bbox_xy`, then
`x = focal*X/Z + princpt` (`_project_to_pixels`, [A3]). If this is off,
triangulation is systematically wrong (the DLT reprojection-error diagnostic
will flag it). **Check**: open `viz/H_000000.png` — the projected body skeleton
must sit on the person. If systematically off, the per-person camera derivation
or bbox (`process_bbox`) differs from your commit; adjust `_project_to_pixels`.

## 9. Known pitfalls / trade-offs

- **⚠️ mmcv / roi_align on Blackwell (the one real blocker).** The SMPLer-X
  **regressor** (not our code) imports `from mmcv.ops.roi_align import roi_align`
  (`common/nets/smpler_x.py`) and `from mmpose.models import build_posenet`
  (`main/SMPLer_X.py`, builds the ViT-H backbone via the bundled
  `transformer_utils`). mmdet is GONE (YOLO does detection), but **mmcv + the
  mmpose fork must stay** — they cannot be removed. mmcv-full 1.7.1 has **no
  prebuilt cu126/torch2.7 wheel**, so on RTX 5080 choose one:
  - **(A) source-build** `mmcv-full==1.7.1` against your torch2.7/cu126/sm_120:
    `MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install mmcv-full==1.7.1 --no-build-isolation`
    (needs the CUDA toolkit; usually works since the roi_align kernel compiles to
    the target arch).
  - **(B) torchvision shim** (no mmcv-full compile): `pip install mmcv` (lite,
    pure-python, any torch) for `mmcv.Config`/`runner`/registry, then make the
    single `roi_align` call resolve to `torchvision.ops.roi_align` (same op,
    Blackwell-native). A 3-line monkeypatch before importing `base` does it;
    ask before enabling — it's a behavior-equivalent drop-in, not a fake stub.
  Decision pending (see ask_human); default requirements pin `mmcv-full==1.7.1`.
- **numpy 2.x** — pin to 1.26.4 (`environment.yml` already does).
- **FOV overlap holes** — 3 head-cams may not all overlap everywhere; body joints
  seen in <2 views fall back to single-view (`source=singleview`) or `missing`.
  This is the inherent trade-off of head-mounted multi-view; per-joint `source`
  is always honest about it.
- **Single-view fallback scale** — without a triangulated anchor that frame,
  single-view body uses body-shape `beta` height (approx) and sits at the H-frame
  origin; absolute translation isn't observable from one view.
- **Dual-env fallback** — if SMPLer-X + its mmpose stack won't coexist with
  something else you need, run body in `pose3d` and (optional) HaMeR in a second
  env; the data contract is the per-view SMPL-X record.
- **HaMeR (optional)** — disabled. To enable: `git clone geopavlakos/hamer` into
  `third_party/hamer`, get `MANO_RIGHT.pkl` from mano.is.tue.mpg.de, set
  `config.yaml: hand.backend=hamer`, then wire `hamer_wrapper.infer_hands()` to
  your pinned HaMeR commit's API.

## 10. Repo layout

```
pose3d/                      # this dir (run from here)
├── run_pipeline.py          # orchestrator
├── config.yaml              # all params (paths, board, fusion, model)
├── environment.yml / requirements.txt / download_models.sh
├── third_party/SMPLer-X/    # cloned upstream (we only wrap it)
└── pose3d/                  # our package
    ├── schema.py            # 78 joint names + SMPL-X index maps + source enum
    ├── body/smplerx_wrapper.py  + body_mapping.py
    ├── hand/smplx_hand_mapping.py + hamer_wrapper.py (optional)
    ├── calib/checkerboard.py + multicam_calib.py
    ├── triangulate/dlt.py
    ├── fuse/view_selector.py + hand_attach.py
    ├── io/time_align.py + video_reader.py + pose_writer.py
    └── viz/skeleton_viewer.py
```
