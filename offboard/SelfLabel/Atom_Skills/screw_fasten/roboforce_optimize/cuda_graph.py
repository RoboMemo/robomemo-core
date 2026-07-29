from __future__ import annotations
import torch
import gc
from typing import Any


class CUDAGraphRunner:
    """CUDA Graph wrapper that captures and replays model inference.

    Eliminates Python-side kernel launch overhead by capturing the entire
    forward pass as a CUDA Graph. Replays the graph for each subsequent call.

    Usage:
        model = YourModel().eval().cuda()
        runner = CUDAGraphRunner(model, warmup_iters=3)
        runner.capture(batch)           # one-time capture
        output = runner.replay(batch)   # fast replay
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device | str = "cuda",
        warmup_iters: int = 3,
        static_input_cb: callable | None = None,
    ):
        self.model = model.eval()
        self.device = torch.device(device)
        self.warmup_iters = warmup_iters
        self.static_input_cb = static_input_cb
        self._graph: torch.cuda.CUDAGraph | None = None
        self._static_inputs: dict[str, Any] = {}
        self._static_outputs: dict[str, Any] = {}
        self._captured = False

    def capture(self, sample_inputs: dict[str, torch.Tensor] | None = None):
        """Capture the model forward pass into a CUDA Graph.

        Args:
            sample_inputs: Example inputs with the same shapes/types as
                           the inputs that will be replayed. If None, the
                           model is called without args to infer static shapes.
        """
        torch.cuda.synchronize()

        # warmup
        if sample_inputs is not None:
            for _ in range(self.warmup_iters):
                self.model(**sample_inputs)
        else:
            for _ in range(self.warmup_iters):
                self.model()

        torch.cuda.synchronize()

        if self.static_input_cb:
            self.static_input_cb(self.model)

        self._graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self._graph):
            if sample_inputs is not None:
                self._static_outputs = self.model(**sample_inputs)
            else:
                self._static_outputs = self.model()

        self._captured = True
        torch.cuda.synchronize()

    def replay(self, inputs: dict[str, torch.Tensor] | None = None) -> Any:
        """Replay the captured CUDA Graph with new inputs.

        Args:
            inputs: Input tensors. Must have the same shapes as the
                    capture inputs. Data is copied into the static
                    memory before replay.

        Returns:
            Model output tensors (same references each call).
        """
        if not self._captured:
            raise RuntimeError("CUDA Graph not captured. Call capture() first.")

        if inputs is not None:
            for key, val in inputs.items():
                if key in self._static_inputs:
                    self._static_inputs[key].copy_(val)
                else:
                    if isinstance(val, torch.Tensor):
                        static = val.detach().clone()
                        self._static_inputs[key] = static

        self._graph.replay()
        torch.cuda.synchronize()
        return self._static_outputs

    @property
    def is_captured(self) -> bool:
        return self._captured

    def reset(self):
        """Release the captured graph and static buffers."""
        self._graph = None
        self._static_inputs.clear()
        if isinstance(self._static_outputs, dict):
            self._static_outputs.clear()
        else:
            self._static_outputs = None
        self._captured = False
        gc.collect()
        torch.cuda.empty_cache()
