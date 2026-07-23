"""
pose3d.io.video_reader — lightweight cv2 video reader.

Reads frames as RGB uint8 HxWx3. Provides random-access by frame index and
downsampled iteration (used by calibration sampling and inference batching).
"""
from __future__ import annotations
import cv2
import numpy as np


class VideoReader:
    def __init__(self, path: str):
        self.path = path
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise IOError(f"VideoReader: cannot open {path}")

    @property
    def fps(self) -> float:
        return float(self.cap.get(cv2.CAP_PROP_FPS))

    @property
    def n_frames(self) -> int:
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return n if n > 0 else 0

    @property
    def width(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def read(self, idx: int) -> np.ndarray | None:
        """Return RGB uint8 frame at index idx, or None if unavailable."""
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, bgr = self.cap.read()
        if not ok or bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def iter_frames(self, step: int = 1, max_frames: int | None = None):
        """Yield (idx, rgb) for every `step`-th frame in order (sequential)."""
        idx = 0
        yielded = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        while True:
            ok, bgr = self.cap.read()
            if not ok:
                break
            if idx % step == 0:
                yield idx, cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                yielded += 1
                if max_frames is not None and yielded >= max_frames:
                    break
            idx += 1

    def sample_at_fps(self, target_fps: float):
        """Yield (frame_idx, t_seconds, rgb) at ~target_fps (temporal samples)."""
        src = self.fps or 59.94
        step = max(1, int(round(src / target_fps)))
        for idx, rgb in self.iter_frames(step=step):
            yield idx, idx / src, rgb

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.release()



