from __future__ import annotations

import gc
from typing import Any

import torch


class CUDAGraphRunner:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
        warmup_iters: int = 3,
    ):
        self.model = model.eval()
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: dict[str, Any] = {}
        self._static_outputs: Any = None
        self._captured = False

    def capture(self, sample_inputs: dict[str, torch.Tensor]):
        torch.cuda.synchronize()
        for _ in range(self.warmup_iters):
            self.model(**sample_inputs)
        torch.cuda.synchronize()

        self._graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self._graph):
            self._static_outputs = self.model(**sample_inputs)
        self._captured = True
        torch.cuda.synchronize()

    def replay(self, inputs: dict[str, torch.Tensor]) -> Any:
        if not self._captured:
            raise RuntimeError("CUDA Graph not captured. Call capture() first.")
        for key, val in inputs.items():
            if key in self._static_inputs:
                self._static_inputs[key].copy_(val)
            else:
                if isinstance(val, torch.Tensor):
                    self._static_inputs[key] = val.detach().clone()
        self._graph.replay()
        torch.cuda.synchronize()
        return self._static_outputs

    @property
    def is_captured(self) -> bool:
        return self._captured

    def reset(self):
        self._graph = None
        self._static_inputs.clear()
        self._static_outputs = None
        self._captured = False
        gc.collect()
        torch.cuda.empty_cache()
