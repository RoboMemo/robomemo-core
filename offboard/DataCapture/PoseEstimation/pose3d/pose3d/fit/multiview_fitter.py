"""
pose3d.fit.multiview_fitter — global multi-view SMPL-X fitting (SCAFFOLD).

Defines ONE SMPL-X(theta, beta) per frame and minimizes multi-view reprojection
+ priors over all cameras. Opt-in via --fusion global_fit (not wired yet).
"""
from __future__ import annotations
import numpy as np

class MultiViewFitter:
    """Global multi-view SMPL-X fitter (scaffold; fitting bodies TODO)."""

    def __init__(self, cfg: dict, smplx_model_dir: str,
                 vposer_dir: str | None = None, device: str = "cuda"):
        """cfg: the `fusion.global_fit` sub-config (weights, optimizer, stages).
        Heavy deps (torch, smplx, smplify-x prior/optimizer, VPoser) are built
        lazily in _build_models(), which is TODO until VPoser + smplify-x land.
        """
        self.cfg = cfg
        self.smplx_model_dir = smplx_model_dir
        self.vposer_dir = vposer_dir
        self.device = device
        self.weights = cfg.get("weights", {})   # {reproj_body, reproj_hand, vposer, ...}
        self._models_built = False
        # self.body_model / self.vposer / self.angle_prior / self.optimizer set in _build_models()

    # ================================================================ build
    def _build_models(self):
        """Build smplx body model + VPoser + AnglePrior + optimizer. TODO.

        Plan (see MULTIVIEW_FITTING_PLAN.md §1,§4):
          import smplx; self.body_model = smplx.create(smplx_model_dir, 'smplx', ...)
          from smplifyx.prior import create_prior
          self.vposer = create_prior('vposer', data_dir=vposer_dir)   # body pose prior
          self.angle_prior = create_prior('angle')                    # joint box limits
          from smplifyx.optimizers import build_optimizer             # LBFGS-LS
        Raises a clear error if smplify-x / VPoser not vendored yet.
        """
        raise NotImplementedError(
            "MultiViewFitter._build_models: TODO — needs smplify-x vendored + "
            "VPoser at self.vposer_dir. See docs/MULTIVIEW_FITTING_PLAN.md §8.")

    # ================================================================ public
    def fit_sequence(self, per_view_keypoints: list, calib: dict,
                     init: list | None = None) -> list:
        """Fit all frames.

        per_view_keypoints: list[frame] of {view: {'body2d':{name:xy},
                                                   'hand2d':{side:{name:xy}},
                                                   'conf':float}}
        calib: calibration.json dict (K_v, dist_v, extrinsics_v).
        init: optional per-frame warm-start from SMPLer-X
              (list[frame] of {global_orient,body_pose,hand_pose,betas,transl}).
        Returns: list[frame] of {'theta','beta','transl','joints3d',
                                 'joints2d_per_view'} -> feeds the same poses.json.

        TODO: downsampling (fit 1/N, interpolate rest), temporal window option,
        per-frame dispatch to fit_frame().
        """
        raise NotImplementedError(
            "MultiViewFitter.fit_sequence: TODO — pending cali re-record + VPoser. "
            "Scaffold only this round. Use --fusion triangulate (default).")

    def fit_frame(self, kp2d_views: dict, calib: dict,
                  init_frame: dict | None = None) -> dict:
        """Single-frame global fit.

        kp2d_views: {view: {'body2d':.., 'hand2d':.., 'conf':..}} for one frame.
        init_frame: warm-start params from SMPLer-X (or None for cold start).
        Returns: {'theta','beta','transl','joints3d','joints2d_per_view'}.

        TODO: warm-start -> LBFGS stage loop -> minimize total_loss().
        """
        raise NotImplementedError("MultiViewFitter.fit_frame: TODO (see plan §3).")

    # ================================================================ losses
    # All loss bodies are TODO; signatures fixed so the impl is a fill-in.
    def total_loss(self, params: dict, kp2d_views: dict, calib: dict,
                   prev_params: dict | None = None) -> float:
        """Σ_v reproj + λ·priors. TODO."""
        raise NotImplementedError("total_loss: TODO.")

    def reproj_loss(self, joints3d: np.ndarray, kp2d_views: dict,
                    Ps: dict, conf: dict, mode: str = "body+hand") -> float:
        """Σ_v ‖Proj_v(joints3d) − Kp2d_v‖², conf-weighted, body and/or hand.
        mode: 'body' | 'hand' | 'body+hand'. TODO (use project_to_views)."""
        raise NotImplementedError("reproj_loss: TODO.")

    def vposer_prior(self, body_pose) -> float:
        """λ1 · BodyPrior(VPoser) on body pose only (NOT hands). TODO."""
        raise NotImplementedError("vposer_prior: TODO.")

    def joint_limit_prior(self, pose, fingers: bool = True) -> float:
        """λ2 · AnglePrior box limits; fingers=True emphasizes MCP/PIP/DIP
        (hands rely on this, not VPoser). TODO."""
        raise NotImplementedError("joint_limit_prior: TODO.")

    def shape_prior(self, betas) -> float:
        """‖β‖². TODO."""
        raise NotImplementedError("shape_prior: TODO.")

    def temporal_smooth(self, pose_t, pose_prev) -> float:
        """λ3 · ‖θ_t − θ_{t-1}‖² (sequence mode only). TODO."""
        raise NotImplementedError("temporal_smooth: TODO.")

    def penetration(self, vertices) -> float:
        """λ4 · CollisionLoss (pytorch3d/bvh self-intersection). TODO.
        Default off (λ4=0) until stable."""
        raise NotImplementedError("penetration: TODO.")

    # ================================================================ helpers
    def warm_start_from_smplerx(self, per_view_records: dict) -> dict:
        """Build a per-frame init from SMPLer-X outputs.

        Strategy (plan §3): pick the view with best det_score (or quaternion-mean
        the 3 views' axis-angles); transl from triangulated pelvis / median cam_trans.
        Returns the init_frame dict for fit_frame().
        TODO.
        """
        raise NotImplementedError("warm_start_from_smplerx: TODO.")

    def project_to_views(self, joints3d_H: np.ndarray, calib: dict) -> dict:
        """Project 3D joints in the H frame to each view's pixel coords.

        CONCRETE (pure projection; reused from the triangulate path):
          for v: P_v = K_v [R_v | t_v], x_v = P_v [X;1], pixel = x_v[:2]/x_v[2]
        (calib extrinsics are stored other->H; build_projection_matrices inverts
        them to H->cam. Here we do the same inversion inline.)
        """
        out = {}
        for v in calib["views"]:
            K = np.asarray(calib["K"][v], float)
            R = np.asarray(calib["extrinsics"][v]["R"], float)
            t = np.asarray(calib["extrinsics"][v]["t"], float).reshape(3, 1)
            R_inv, t_inv = R.T, -R.T @ t          # H -> cam
            P = K @ np.hstack([R_inv, t_inv])
            Xh = np.hstack([joints3d_H, np.ones((joints3d_H.shape[0], 1))])
            x = Xh @ P.T
            z = np.where(np.abs(x[:, 2]) < 1e-9, 1e-9, x[:, 2])
            out[v] = x[:, :2] / z[:, None]
        return out

