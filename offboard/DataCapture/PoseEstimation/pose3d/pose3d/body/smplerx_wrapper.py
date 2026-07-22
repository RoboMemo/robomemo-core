"""
pose3d.body.smplerx_wrapper — adapter around the REAL upstream SMPLer-X.

Verified against caizhongang/SMPLer-X `main/inference.py` (raw source):
  - entry: `from config import cfg` (yacs) -> `from base import Demoer`
            `demoer = Demoer(); demoer._make_model(); demoer.model.eval()`
  - person detect (OUR swap): ultralytics YOLO (yolov8n.pt, class 0 = person),
            largest box. Upstream uses mmdet faster_rcnn, but mmdet has no
            torch2.6/cu124/Blackwell wheel, so detection is YOLO. Downstream
            crop/regressor is the upstream path, unchanged.
  - crop: `utils.preprocessing.process_bbox` + `generate_patch_image` -> 224 patch
  - forward: `out = demoer.model({'img':...}, {}, {}, 'test')`
  - outputs: smplx_root_pose / smplx_body_pose / smplx_lhand_pose / smplx_rhand_pose /
             smplx_shape(betas) / smplx_expr / cam_trans / smplx_mesh_cam
  - projection: PERSPECTIVE (NOT weak-persp pred_cam):
        focal   = cfg.focal / cfg.input_body_shape * bbox_wh
        princpt = cfg.princpt / cfg.input_body_shape * bbox_wh + bbox_xy
        x = focal[0]*X/Z + princpt[0]   (cam-frame mesh already has cam_trans applied)

REGRESSOR STILL NEEDS mmcv/mmpose: `common/nets/smpler_x.py` does
`from mmcv.ops.roi_align import roi_align` and `main/SMPLer_X.py` does
`from mmpose.models import build_posenet` (+ `from mmcv import Config`). Only
the DETECTOR (mmdet) was removed; mmcv-full + the bundled transformer_utils
mmpose fork remain required — see README §9 for the Blackwell path.

UPSTREAM COUPLING — ISOLATED & CLEARLY MARKED
  Adapter points (edit ONLY if you pin a different commit and the API drifted):
    [A1] _import_regressor()  — sys.path, cfg load, Demoer build, YOLO load
    [A2] _raw_inference(rgb)  — YOLO detect -> crop -> demoer.model -> params + cam
    [A3] _project_to_pixels() — perspective focal/princpt projection
  Everything downstream (smplx forward -> joints/mesh, body mapping, schema) is ours
  and stable (joint indices verified against vchoutas/smplx, see CODE_REVIEW 🟢-13).

MAC: cannot run this (no CUDA). On Linux, FIRST run a single frame and print
`out.keys()` + shapes to confirm before trusting outputs (see README §8 VERIFY).
"""
from __future__ import annotations
import os
import sys
import numpy as np

from .body_mapping import extract_body
from ..schema import BODY_JOINT_NAMES


class SMPLerXWrapper:
    def __init__(self, cfg: dict, smplx_model_dir: str, device: str = "cuda"):
        """cfg: the 'smplerx' sub-config from config.yaml."""
        import torch  # heavy; import here so the module imports without torch
        self.torch = torch
        self.cfg = cfg
        self.device = device
        self.smplx_model_dir = smplx_model_dir
        self._smplx_model = None
        self.regressor = self._import_regressor()      # [A1] adapter

    # =============================================================== [A1] ===
    def _import_regressor(self):
        repo = self.cfg["repo_dir"]
        if not os.path.isdir(repo):
            raise FileNotFoundError(
                f"SMPLer-X repo not found at {repo}. Run download_models.sh.")
        # upstream inference.py inserts main/data/common onto sys.path
        for sub in ("main", "data", "common"):
            p = os.path.join(repo, sub)
            if os.path.isdir(p) and p not in sys.path:
                sys.path.insert(0, p)

        # --- yacs config ---
        from config import cfg as smx_cfg  # type: ignore
        config_path = self.cfg.get("config", "main/config/config_smpler_x_h32.py")
        if not os.path.isabs(config_path):
            config_path = os.path.join(repo, config_path)
        ckpt = self.cfg.get("checkpoint", "pretrained_models/smpler_x_h32_correct.pth.tar")
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(repo, ckpt)
        smx_cfg.get_config_fromfile(config_path)
        smx_cfg.update_test_config("na", "na", shapy_eval_split=None,
                                   pretrained_model_path=ckpt, use_cache=False)
        smx_cfg.update_config(num_gpus=1, exp_name="pose3d_infer")
        # point SMPLer-X at our already-provided SMPL-X body models
        hmf = os.path.join(repo, "common", "utils", "human_model_files")
        os.makedirs(os.path.join(hmf, "smplx"), exist_ok=True)
        try:
            # symlink our npz dir into the expected layout (idempotent)
            link = os.path.join(hmf, "smplx")
            if not os.path.lexists(link) or os.path.islink(link):
                if os.path.islink(link):
                    os.remove(link)
                os.symlink(os.path.abspath(self.smplx_model_dir), link)
        except OSError:
            pass  # symlink may fail on some FS; download_models.sh also sets it
        self.smx_cfg = smx_cfg

        # --- build the regressor (Demoer) ---
        from base import Demoer  # type: ignore
        demoer = Demoer()
        demoer._make_model()
        demoer.model.eval()

        # --- person detector: ultralytics YOLO (drops the mmdet/mmcv-full
        #     CUDA-op dependency that has no torch2.6/cu124/Blackwell wheel).
        #     Downstream (process_bbox -> generate_patch_image -> demoer.model)
        #     is UNCHANGED; only the detector is swapped.
        from ultralytics import YOLO  # type: ignore
        yolo_w = self.cfg.get("detector", "yolov8n.pt")
        self.detector = YOLO(yolo_w)
        self.det_conf = float(self.cfg.get("det_conf", 0.5))

        # --- preprocessing helpers (upstream crop utils) ---
        from utils.preprocessing import process_bbox, generate_patch_image  # type: ignore
        self._process_bbox = process_bbox
        self._generate_patch_image = generate_patch_image
        return demoer

    # =============================================================== [A2] ===
    def _raw_inference(self, rgb: np.ndarray):
        """rgb: HxWx3 uint8 RGB. Returns normalized param dict or None (no person)."""
        import torchvision.transforms as transforms

        H, W = rgb.shape[:2]
        bgr = rgb[:, :, ::-1].copy()   # YOLO/cv2 work in BGR

        # 1. person detection (ultralytics YOLO, class 0 = person)
        res = self.detector(bgr, conf=self.det_conf, classes=[0], verbose=False)[0]
        bb = getattr(res, "boxes", None)
        if bb is None or len(bb) == 0:
            return None
        xyxy = bb.xyxy.cpu().numpy()      # (N,4)
        conf = bb.conf.cpu().numpy()      # (N,)
        areas = np.clip(xyxy[:, 2] - xyxy[:, 0], 0, None) * np.clip(xyxy[:, 3] - xyxy[:, 1], 0, None)
        i = int(areas.argmax())           # largest person box
        x1, y1, x2, y2 = xyxy[i]
        score = float(conf[i])
        if (x2 - x1) < 1 or (y2 - y1) < 1:
            return None
        xywh = np.array([x1, y1, x2 - x1, y2 - y1], dtype=np.float32)

        # 2. crop to 224 patch (upstream preprocessing)
        bbox = self._process_bbox(xywh, W, H)
        patch, _img2bb, _bb2img = self._generate_patch_image(
            bgr, bbox, 1.0, 0.0, False, self.smx_cfg.input_img_shape)
        img = transforms.ToTensor()(patch.astype(np.float32)) / 255.0
        img = img.to(self.device)[None, :, :, :]

        # 3. regressor forward
        with self.torch.no_grad():
            out = self.regressor.model({"img": img}, {}, {}, "test")

        def flat(key, n3=None):
            v = out[key]
            v = v.detach().cpu().numpy().reshape(-1)
            return v

        # 4. camera (perspective) for this person, from bbox (see inference.py)
        ibs = self.smx_cfg.input_body_shape          # [H_crop, W_crop] e.g. [224,224]
        focal = [self.smx_cfg.focal[0] / ibs[1] * bbox[2],
                 self.smx_cfg.focal[1] / ibs[0] * bbox[3]]
        princpt = [self.smx_cfg.princpt[0] / ibs[1] * bbox[2] + bbox[0],
                   self.smx_cfg.princpt[1] / ibs[0] * bbox[3] + bbox[1]]

        return {
            "global_orient":   flat("smplx_root_pose"),
            "body_pose":       flat("smplx_body_pose"),
            "left_hand_pose":  flat("smplx_lhand_pose"),
            "right_hand_pose": flat("smplx_rhand_pose"),
            "betas":           flat("smplx_shape"),
            "cam_trans":       flat("cam_trans"),
            "focal": focal, "princpt": princpt,
            "body_bbox": np.array([x1, y1, x2, y2], dtype=np.float64),
            "det_score": float(score),
        }

    # ----------------------------------------------------------------- #
    # smplx forward -> joints + mesh (single person), root-centered.
    # cam_trans is applied additively for the camera-frame projection.
    # ----------------------------------------------------------------- #
    def _smplx_forward(self, p: dict):
        import smplx  # vchoutas/smplx; indices verified (CODE_REVIEW 🟢-13)
        if self._smplx_model is None:
            self._smplx_model = smplx.create(
                self.smplx_model_dir, model_type="smplx",
                gender=self.cfg.get("gender", "neutral"), ext="npz",
                batch_size=1, use_pca=False, create_global_orient=False,
                create_hand_pose=False).to(self.device)

        def _aa(v, j):
            """axis-angle (j*3,) -> (1, j, 3) tensor."""
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            return self.torch.as_tensor(v, device=self.device).view(1, max(1, v.size // 3), 3)

        m = self._smplx_model(
            global_orient=_aa(p["global_orient"], 1),
            body_pose=_aa(p["body_pose"], max(1, len(p["body_pose"]) // 3)),
            betas=self.torch.as_tensor(np.asarray(p["betas"], np.float32).reshape(1, -1),
                                       device=self.device),
            left_hand_pose=_aa(p["left_hand_pose"], 15),
            right_hand_pose=_aa(p["right_hand_pose"], 15),
            return_verts=True)
        joints = m.joints.detach().cpu().numpy()[0]      # (J,3) root-centered
        verts = m.vertices.detach().cpu().numpy()[0]     # (V,3) root-centered
        return joints, verts

    # =============================================================== [A3] ===
    def _project_to_pixels(self, joints_root: np.ndarray, cam_trans: np.ndarray,
                           focal, princpt) -> np.ndarray:
        """Perspective projection (matches upstream render_mesh)."""
        cam = np.asarray(cam_trans, float).reshape(3)
        P = joints_root + cam                        # camera-frame (root + cam_trans)
        Z = P[:, 2]
        Z = np.where(np.abs(Z) < 1e-6, 1e-6, Z)
        u = focal[0] * P[:, 0] / Z + princpt[0]
        v = focal[1] * P[:, 1] / Z + princpt[1]
        return np.stack([u, v], 1)

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    def infer_frame(self, rgb: np.ndarray) -> dict:
        """rgb: HxWx3 uint8. Returns a per-view pose record (or empty if none)."""
        p = self._raw_inference(rgb)
        if p is None or p["global_orient"] is None or p["body_pose"] is None:
            return self._empty()
        joints3d, verts = self._smplx_forward(p)         # root-centered (J,3),(V,3)
        body3d = extract_body(joints3d)
        proj2d = self._project_to_pixels(joints3d, p["cam_trans"], p["focal"], p["princpt"])
        body2d = {n: proj2d[i] for i, n in enumerate(BODY_JOINT_NAMES)}
        return {
            "has_person": True, "view": None,
            "det_score": float(p["det_score"]),
            "betas": (p["betas"].copy() if p["betas"] is not None else None),
            "cam_trans": (p["cam_trans"].copy() if p["cam_trans"] is not None else None),
            "body_bbox": (p["body_bbox"].copy() if p["body_bbox"] is not None else None),
            "joints3d_smplx": joints3d.copy(),     # root-centered (for Procrustes)
            "verts_smplx": verts.copy(),           # root-centered mesh (fingertips)
            "body_joints3d": body3d,               # root-centered
            "body_joints2d": body2d,               # projected to original-image pixels
        }

    @staticmethod
    def _empty():
        return {"has_person": False, "view": None,
                "body_joints3d": {}, "body_joints2d": {},
                "joints3d_smplx": None, "verts_smplx": None,
                "betas": None, "det_score": 0.0}
