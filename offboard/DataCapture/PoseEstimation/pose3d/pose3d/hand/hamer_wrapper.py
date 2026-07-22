"""
pose3d.hand.hamer_wrapper — OPTIONAL hand backend (disabled by default).

Enable only if SMPLer-X hand precision is insufficient AND you have obtained
MANO_RIGHT.pkl (https://mano.is.tue.mpg.de/, registration). The main line uses
SMPLer-X hands (hand/smplx_hand_mapping.py); this module is not imported unless
config.yaml sets hand.backend=hamer.

HaMeR regresses MANO -> hand mesh -> 21 keypoints (incl. fingertips), which map
to 21 of our 27 names directly (source=hamer); *_Extra + Palm_Center are derived
(DESIGN.md §4.1). Output contract matches smplx_hand_mapping.map_hand.
"""
from __future__ import annotations
import os
import sys
import numpy as np

from ..schema import HAND_JOINTS_BASE, FINGER_SEGMENTS, PALM_MCP_NAMES, Source

# MANO 21-keypoint order -> our base name (1:1 for 21 of the 27)
MANO21_TO_NAME = [
    "Wrist",
    "Thumb_CMC", "Thumb_MCP", "Thumb_IP", "Thumb_Tip",
    "Index_MCP", "Index_PIP", "Index_DIP", "Index_Tip",
    "Middle_MCP", "Middle_PIP", "Middle_DIP", "Middle_Tip",
    "Ring_MCP", "Ring_PIP", "Ring_DIP", "Ring_Tip",
    "Pinky_MCP", "Pinky_PIP", "Pinky_DIP", "Pinky_Tip",
]
EXTRA_RATIO = 0.5


class HaMeRWrapper:
    def __init__(self, cfg: dict, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        repo = cfg["repo_dir"]
        mano = cfg.get("mano_path")
        if not os.path.isdir(repo):
            raise FileNotFoundError(f"HaMeR repo not found at {repo}; it is OPTIONAL.")
        if not mano or not os.path.isfile(mano):
            raise FileNotFoundError(
                f"MANO_RIGHT.pkl not found at {mano}. Download from mano.is.tue.mpg.de. "
                f"HaMeR is optional; use hand.backend=smplx (default) instead.")
        if repo not in sys.path:
            sys.path.insert(0, os.path.abspath(repo))
        # Build the HaMeR predictor. Import path may vary across commits.
        from hamer.configs import get_config  # type: ignore
        from hamer.models import load_hamer    # type: ignore
        ckpt = cfg.get("checkpoint")
        model_cfg = get_config()
        self.predictor = load_hamer(model_cfg, ckpt or "").to(device).eval()
        self.model_cfg = model_cfg

    def infer_hands(self, rgb: np.ndarray) -> dict:
        """Run HaMeR on one RGB frame. Returns {"L": mano21(21,3), "R": ...}.

        The exact HaMeR batched-inference API varies by commit; adapt the body
        of this method to your pinned hamer/ commit. It should return per-hand
        3D keypoints in a ROOT-CENTERED frame (HaMeR's own scale).
        """
        raise NotImplementedError(
            "Wire infer_hands() to your pinned HaMeR commit's inference API. "
            "See hamer/demo.py. Return {'L': (21,3), 'R': (21,3)} root-centered.")

    def map_hand(self, side: str, mano21: np.ndarray) -> dict:
        """MANO-21 (21,3) -> our 27 names/side. Tips are source=hamer."""
        out = {}
        for i, name in enumerate(MANO21_TO_NAME):
            out[name] = {"xyz": np.asarray(mano21[i], float).copy(),
                         "conf": 1.0, "source": Source.HAMER.value}
        # *_Extra = midpoint of PIP-DIP phalanx (derived)
        for finger in ("Thumb", "Index", "Middle", "Ring", "Pinky"):
            seg = FINGER_SEGMENTS[finger]
            a = out[seg[1]]["xyz"]; b = out[seg[2]]["xyz"]
            out[f"{finger}_Extra"] = {"xyz": (1 - EXTRA_RATIO) * a + EXTRA_RATIO * b,
                                       "conf": 1.0, "source": Source.DERIVED.value}
        # Palm_Center = centroid of Wrist + 4 MCPs (derived)
        pts = [out["Wrist"]["xyz"]] + [out[f"{m}_MCP"]["xyz"] for m in PALM_MCP_NAMES]
        out["Palm_Center"] = {"xyz": np.mean(np.stack(pts, 0), 0),
                              "conf": 1.0, "source": Source.DERIVED.value}
        assert set(out) == set(HAND_JOINTS_BASE)
        return out
