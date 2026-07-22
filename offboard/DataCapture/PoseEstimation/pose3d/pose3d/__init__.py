"""pose3d — 3D whole-body pose pipeline (SMPLer-X body+hands + multi-view DLT).

Run from this directory (PoseEstimation/pose3d/):
    python run_pipeline.py --recording 0721-1
so that the inner `pose3d` package is importable.
"""
from . import schema  # noqa: F401

__version__ = "1.0.0"
