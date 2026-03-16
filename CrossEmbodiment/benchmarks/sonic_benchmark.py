#!/usr/bin/env python3
"""
SONIC Retarget Inference Benchmark Suite.

Comprehensive benchmarks for real-time deployment readiness on humanoid robots.
Measures latency, throughput, memory, stability, and per-robot comparison.

Target: 50 Hz control loop → 20 ms total budget per step.

Usage:
    python3 -m cross_embodiment_retarget_demo.benchmarks.sonic_benchmark
    python3 cross_embodiment_retarget_demo/benchmarks/sonic_benchmark.py
    python3 cross_embodiment_retarget_demo/benchmarks/sonic_benchmark.py --iterations 5000 --warmup 200
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import platform
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root and ensure importability
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from cross_embodiment_retarget_demo.src.body_types import BodyPose, InputSource, Pose7D
from cross_embodiment_retarget_demo.src.demo_runner import load_config
from cross_embodiment_retarget_demo.src.isaac_env import MockPhysicsEnv
from cross_embodiment_retarget_demo.src.mock_motion import MockMotionSource
from cross_embodiment_retarget_demo.src.sonic_retarget import (
    MockSonicBackend,
    SonicRetarget,
    _get_robot_cfg,
)

# Optional deps ─────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROBOTS: List[Tuple[str, int]] = [
    ("unitree_g1", 29),
    ("unitree_h1", 19),
    ("fourier_gr1t2", 32),
]
TARGET_HZ = 50
TARGET_DT_MS = 1000.0 / TARGET_HZ  # 20 ms


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt_ms(ms: float) -> str:
    if ms < 0.001:
        return f"{ms * 1000:.2f}µs"
    if ms < 1.0:
        return f"{ms:.4f}ms"
    return f"{ms:.2f}ms"


def _fmt_fps(fps: float) -> str:
    if fps >= 1_000_000:
        return f"{fps / 1_000_000:.2f}M"
    if fps >= 1_000:
        return f"{fps:,.0f}"
    return f"{fps:,.0f}"


def _rss_mb() -> float:
    if psutil is not None:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _gpu_mem_mb() -> Tuple[float, float]:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader", "--id=0"],
            text=True, timeout=5,
        ).strip()
        used, total = out.split(",")
        return float(used.strip()), float(total.strip())
    except Exception:
        return 0.0, 0.0


def _gpu_name() -> str:
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            text=True, timeout=5,
        ).strip()
        return out
    except Exception:
        return "N/A"


def _make_body_pose(t: float = 0.0) -> BodyPose:
    freq = 1.0
    phase = 2 * math.pi * freq * t
    arm_swing = 0.15 * math.sin(phase)
    stride = 0.15 * math.sin(phase)
    foot_lift = max(0.0, 0.05 * math.sin(phase))
    foot_lift_r = max(0.0, 0.05 * math.sin(phase + math.pi))
    return BodyPose(
        head=Pose7D(np.array([0, 0, 1.6 + 0.02 * math.sin(2 * phase)], dtype=np.float32),
                    np.array([1, 0, 0, 0], dtype=np.float32)),
        left_hand=Pose7D(np.array([-0.25, -arm_swing, 0.95], dtype=np.float32),
                         np.array([1, 0, 0, 0], dtype=np.float32)),
        right_hand=Pose7D(np.array([0.25, arm_swing, 0.95], dtype=np.float32),
                          np.array([1, 0, 0, 0], dtype=np.float32)),
        left_ankle=Pose7D(np.array([-0.1, stride, foot_lift], dtype=np.float32),
                          np.array([1, 0, 0, 0], dtype=np.float32)),
        right_ankle=Pose7D(np.array([0.1, -stride, foot_lift_r], dtype=np.float32),
                           np.array([1, 0, 0, 0], dtype=np.float32)),
        waist=Pose7D(np.array([0, 0, 0.9], dtype=np.float32),
                     np.array([1, 0, 0, 0], dtype=np.float32)),
        source=InputSource.MOCK,
        has_leg_tracking=True,
        has_waist_tracking=True,
    )


def _latency_stats(times_ns: List[int]) -> Dict[str, float]:
    ms = np.array(times_ns, dtype=np.float64) / 1_000_000.0
    return {
        "mean_ms": float(np.mean(ms)),
        "median_ms": float(np.median(ms)),
        "p95_ms": float(np.percentile(ms, 95)),
        "p99_ms": float(np.percentile(ms, 99)),
        "min_ms": float(np.min(ms)),
        "max_ms": float(np.max(ms)),
        "std_ms": float(np.std(ms)),
        "count": len(ms),
    }


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------
@dataclass
class LatencyResult:
    stage: str
    stats: Dict[str, float] = field(default_factory=dict)

@dataclass
class ThroughputResult:
    label: str
    fps: float = 0.0
    infer_per_sec: float = 0.0
    batch_size: int = 1

@dataclass
class PipelineResult:
    stage: str
    mean_ms: float = 0.0

@dataclass
class RobotResult:
    robot: str
    dof: int = 0
    mean_ms: float = 0.0
    fps: float = 0.0

@dataclass
class StabilityResult:
    total: int = 0
    spikes: int = 0
    spike_pct: float = 0.0
    max_spike_ms: float = 0.0
    jitter_ms: float = 0.0
    median_ms: float = 0.0

@dataclass
class MemoryResult:
    rss_before_mb: float = 0.0
    rss_after_mb: float = 0.0
    gpu_before_mb: float = 0.0
    gpu_after_mb: float = 0.0
    gpu_peak_mb: float = 0.0
    gpu_total_mb: float = 0.0

@dataclass
class BenchmarkResults:
    timestamp: str = ""
    gpu_name: str = ""
    backend: str = ""
    python_version: str = ""
    ort_version: str = ""
    numpy_version: str = ""
    os_info: str = ""
    latency: List[Dict[str, Any]] = field(default_factory=list)
    throughput: List[Dict[str, Any]] = field(default_factory=list)
    pipeline: List[Dict[str, Any]] = field(default_factory=list)
    per_robot: List[Dict[str, Any]] = field(default_factory=list)
    stability: Dict[str, Any] = field(default_factory=dict)
    memory: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pretty-printing
# ---------------------------------------------------------------------------
W = 70  # box inner width


def _print_banner(gpu: str, backend: str, robot: str) -> None:
    print()
    print("╔" + "═" * W + "╗")
    print("║" + "SONIC Retarget Inference Benchmark".center(W) + "║")
    info = f"GPU: {gpu} | Backend: {backend} | Robot: {robot}"
    if len(info) > W:
        info = info[: W - 1] + "…"
    print("║" + info.center(W) + "║")
    print("╠" + "═" * W + "╣")


def _section(title: str, sub: str = "") -> None:
    print()
    full = title
    if sub:
        full += f" ({sub})"
    print("┌" + "─" * W + "┐")
    print(f"│ {full}".ljust(W + 1) + "│")
    print("├" + "─" * W + "┤")


def _print_latency_table(results: List[LatencyResult]) -> None:
    hdr = f"│ {'Stage':<32s}│{'Mean':>9s} │{'Median':>9s} │{'P95':>9s} │{'P99':>9s} │{'Max':>9s} │"
    print(hdr)
    sep = "├" + "─" * 33 + "┼" + ("──────────┼" * 5)[:-1] + "┤"
    print(sep)
    for r in results:
        s = r.stats
        print(
            f"│ {r.stage:<32s}"
            f"│{_fmt_ms(s['mean_ms']):>9s} "
            f"│{_fmt_ms(s['median_ms']):>9s} "
            f"│{_fmt_ms(s['p95_ms']):>9s} "
            f"│{_fmt_ms(s['p99_ms']):>9s} "
            f"│{_fmt_ms(s['max_ms']):>9s} │"
        )
    print("└" + "─" * 33 + "┴" + ("──────────┴" * 5)[:-1] + "┘")


def _print_throughput_table(results: List[ThroughputResult]) -> None:
    for r in results:
        fps_s = f"{_fmt_fps(r.fps)} fps"
        print(f"│ {r.label:<38s}│ {fps_s:>29s} │")
    if results:
        seq_fps = results[0].fps
        headroom = seq_fps / TARGET_HZ if TARGET_HZ > 0 else 0
        icon = "✅" if headroom > 2 else "⚠️"
        hl = f"{icon} {headroom:,.0f}x headroom"
        print("├" + "─" * 39 + "┼" + "─" * 30 + "┤")
        print(f"│ {'Target: ' + str(TARGET_HZ) + 'Hz real-time':<38s}│ {hl:>29s} │")
    print("└" + "─" * 39 + "┴" + "─" * 30 + "┘")


def _print_pipeline_table(results: List[PipelineResult]) -> None:
    if not results:
        return
    total_ms = results[0].mean_ms
    margin = TARGET_DT_MS / total_ms if total_ms > 0 else float("inf")
    for i, r in enumerate(results):
        prefix = "├─ " if i > 0 else ""
        name = prefix + r.stage
        extra = ""
        if i == 0:
            extra = f", target {TARGET_DT_MS:.0f}ms = {margin:.0f}x margin"
        val = f"{_fmt_ms(r.mean_ms)} (mean){extra}"
        print(f"│ {name:<28s}│ {val:<40s}│")
    print("└" + "─" * 29 + "┴" + "─" * 41 + "┘")


def _print_robot_table(results: List[RobotResult]) -> None:
    print(f"│ {'Robot':<18s}│{'DOF':>6s} │{'Mean(ms)':>12s} │{'FPS':>14s} │")
    print("├" + "─" * 19 + "┼" + "─" * 7 + "┼" + "─" * 13 + "┼" + "─" * 15 + "┤")
    for r in results:
        print(f"│ {r.robot:<18s}│{r.dof:>6d} │{r.mean_ms:>11.4f} │{_fmt_fps(r.fps):>14s} │")
    print("└" + "─" * 19 + "┴" + "─" * 7 + "┴" + "─" * 13 + "┴" + "─" * 15 + "┘")


def _print_stability_table(r: StabilityResult) -> None:
    rows = [
        ("Latency spikes (>2x median)", f"{r.spikes} ({r.spike_pct:.2f}%)"),
        ("Max spike", _fmt_ms(r.max_spike_ms)),
        ("Jitter (std of Δlatency)", _fmt_ms(r.jitter_ms)),
        ("Median latency", _fmt_ms(r.median_ms)),
    ]
    for label, val in rows:
        print(f"│ {label:<38s}│ {val:>29s} │")
    print("└" + "─" * 39 + "┴" + "─" * 30 + "┘")


def _print_memory_table(m: MemoryResult) -> None:
    rows = [
        ("RSS before model load", f"{m.rss_before_mb:.1f} MB"),
        ("RSS after model load", f"{m.rss_after_mb:.1f} MB"),
        ("GPU memory (before)", f"{m.gpu_before_mb:.0f} MB"),
        ("GPU memory (after model load)", f"{m.gpu_after_mb:.0f} MB"),
        ("GPU peak during inference", f"{m.gpu_peak_mb:.0f} MB"),
        ("GPU total", f"{m.gpu_total_mb:.0f} MB"),
    ]
    for label, val in rows:
        print(f"│ {label:<38s}│ {val:>29s} │")
    print("└" + "─" * 39 + "┴" + "─" * 30 + "┘")


# ---------------------------------------------------------------------------
# Core benchmark class
# ---------------------------------------------------------------------------
class SonicBenchmark:
    """Comprehensive SONIC inference benchmark suite."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        iterations: int = 10000,
        warmup: int = 500,
        sustained_duration: float = 10.0,
    ):
        self.cfg = load_config(config_path)
        self.iterations = iterations
        self.warmup = warmup
        self.sustained_duration = sustained_duration
        self.results = BenchmarkResults()

        # Pin to single CPU core for stable measurements
        try:
            os.sched_setaffinity(0, {0})
            logger.info("Pinned to CPU core 0")
        except (AttributeError, OSError):
            pass

    # ── 1. Inference Latency ──────────────────────────────────

    def bench_inference_latency(self, robot_type: str = "unitree_g1") -> List[LatencyResult]:
        self.cfg["robot"]["type"] = robot_type
        robot_cfg = _get_robot_cfg(self.cfg)
        num_joints = robot_cfg.get("num_joints", 29)

        backend = MockSonicBackend()
        backend.init(self.cfg)
        retarget = SonicRetarget(self.cfg)

        total = self.warmup + self.iterations
        poses = [_make_body_pose(t * 0.02) for t in range(total)]
        prop = np.zeros(num_joints * 2 + 3, dtype=np.float32)

        results: List[LatencyResult] = []

        # ── Proprioception pack ──
        jp = np.zeros(num_joints, dtype=np.float32)
        jv = np.zeros(num_joints, dtype=np.float32)
        gv = np.array([0, 0, -9.81], dtype=np.float32)
        times: List[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            _ = np.concatenate([jp, jv, gv])
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                times.append(t1 - t0)
        results.append(LatencyResult("Proprioception pack", _latency_stats(times)))

        # ── Backend.infer (raw analytical IK) ──
        times = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            _ = backend.infer(poses[i], prop)
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                times.append(t1 - t0)
        results.append(LatencyResult("Backend.infer (raw IK)", _latency_stats(times)))

        # ── Post-processing (vel limit + clamp) ──
        pos_min = robot_cfg["joint_limits"]["position_min"]
        pos_max = robot_cfg["joint_limits"]["position_max"]
        vel_scale = self.cfg["safety"].get("max_joint_velocity_scale", 0.8)
        prev = np.zeros(num_joints, dtype=np.float32)
        times = []
        for i in range(total):
            raw = backend.infer(poses[i % len(poses)], prop)
            t0 = time.perf_counter_ns()
            max_d = vel_scale * 10.0 * 0.02
            delta = np.clip(raw - prev, -max_d, max_d)
            tgt = prev + delta
            tgt = np.clip(tgt, pos_min, pos_max)
            prev = tgt
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                times.append(t1 - t0)
        results.append(LatencyResult("Post-processing", _latency_stats(times)))

        # ── End-to-end (SonicRetarget.infer) ──
        times = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            _ = retarget.infer(poses[i])
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                times.append(t1 - t0)
        results.append(LatencyResult("End-to-end (SonicRetarget.infer)", _latency_stats(times)))

        return results

    # ── 2. Throughput ─────────────────────────────────────────

    def bench_throughput(self, robot_type: str = "unitree_g1") -> List[ThroughputResult]:
        self.cfg["robot"]["type"] = robot_type
        retarget = SonicRetarget(self.cfg)
        results: List[ThroughputResult] = []
        pose = _make_body_pose(0.5)

        # warmup
        for _ in range(self.warmup):
            retarget.infer(pose)

        # Sequential
        t0 = time.perf_counter_ns()
        for _ in range(self.iterations):
            retarget.infer(pose)
        ns = time.perf_counter_ns() - t0
        seq_fps = self.iterations / (ns / 1e9) if ns > 0 else 0
        results.append(ThroughputResult("Sequential FPS", fps=seq_fps))

        # Sustained
        count = 0
        t_s = time.perf_counter()
        t_e = t_s + self.sustained_duration
        while time.perf_counter() < t_e:
            retarget.infer(_make_body_pose(count * 0.02))
            count += 1
        dur = time.perf_counter() - t_s
        sus_fps = count / dur if dur > 0 else 0
        results.append(ThroughputResult(f"Sustained FPS ({self.sustained_duration:.0f}s)", fps=sus_fps))

        # Batch sizes
        for bs in [1, 4, 8, 16, 32]:
            batch_poses = [_make_body_pose(j * 0.02) for j in range(bs)]
            iters = max(100, self.iterations // bs)
            for _ in range(50):
                for p in batch_poses:
                    retarget.infer(p)
            t0 = time.perf_counter_ns()
            for _ in range(iters):
                for p in batch_poses:
                    retarget.infer(p)
            ns = time.perf_counter_ns() - t0
            total_infers = iters * bs
            ips = total_infers / (ns / 1e9) if ns > 0 else 0
            results.append(ThroughputResult(f"Batch={bs}", fps=ips, infer_per_sec=ips, batch_size=bs))

        return results

    # ── 3. Full Pipeline ──────────────────────────────────────

    def bench_full_pipeline(self, robot_type: str = "unitree_g1") -> List[PipelineResult]:
        self.cfg["robot"]["type"] = robot_type
        self.cfg["simulation"]["backend"] = "mock_physics"
        self.cfg["simulation"]["mock_physics"]["enable_visualization"] = False

        retarget = SonicRetarget(self.cfg)
        sim = MockPhysicsEnv()
        sim.init(self.cfg)
        motion = MockMotionSource(self.cfg)
        motion.start()

        buckets: Dict[str, List[int]] = {
            "Input capture": [], "SONIC infer": [],
            "Sim step": [], "Proprioception update": [],
            "Per-step total": [],
        }

        # warmup
        for _ in range(self.warmup):
            bp = motion.latest_pose
            jt = retarget.infer(bp)
            sim.step(jt)
            retarget.update_proprioception(sim.get_joint_pos(), sim.get_joint_vel(), sim.get_gravity_vector())

        for _ in range(self.iterations):
            t_all = time.perf_counter_ns()

            t0 = time.perf_counter_ns()
            bp = motion.latest_pose
            t1 = time.perf_counter_ns()
            buckets["Input capture"].append(t1 - t0)

            t0 = time.perf_counter_ns()
            jt = retarget.infer(bp)
            t1 = time.perf_counter_ns()
            buckets["SONIC infer"].append(t1 - t0)

            t0 = time.perf_counter_ns()
            sim.step(jt)
            t1 = time.perf_counter_ns()
            buckets["Sim step"].append(t1 - t0)

            t0 = time.perf_counter_ns()
            retarget.update_proprioception(sim.get_joint_pos(), sim.get_joint_vel(), sim.get_gravity_vector())
            t1 = time.perf_counter_ns()
            buckets["Proprioception update"].append(t1 - t0)

            buckets["Per-step total"].append(time.perf_counter_ns() - t_all)

        motion.stop()
        sim.close()

        order = ["Per-step total", "Input capture", "SONIC infer", "Sim step", "Proprioception update"]
        return [PipelineResult(stage=s, mean_ms=_latency_stats(buckets[s])["mean_ms"]) for s in order]

    # ── 4. Per-Robot Comparison ───────────────────────────────

    def bench_per_robot(self) -> List[RobotResult]:
        results: List[RobotResult] = []
        for robot_type, dof in ROBOTS:
            self.cfg["robot"]["type"] = robot_type
            retarget = SonicRetarget(self.cfg)
            pose = _make_body_pose(0.5)

            for _ in range(self.warmup):
                retarget.infer(pose)

            times: List[int] = []
            for i in range(self.iterations):
                t0 = time.perf_counter_ns()
                retarget.infer(_make_body_pose(i * 0.02))
                t1 = time.perf_counter_ns()
                times.append(t1 - t0)

            s = _latency_stats(times)
            fps = 1000.0 / s["mean_ms"] if s["mean_ms"] > 0 else float("inf")
            results.append(RobotResult(robot=robot_type, dof=dof, mean_ms=s["mean_ms"], fps=fps))
        return results

    # ── 5. Memory ─────────────────────────────────────────────

    def bench_memory(self, robot_type: str = "unitree_g1") -> MemoryResult:
        gc.collect()
        rss_before = _rss_mb()
        gpu_before, gpu_total = _gpu_mem_mb()

        self.cfg["robot"]["type"] = robot_type
        retarget = SonicRetarget(self.cfg)
        rss_after = _rss_mb()
        gpu_after, _ = _gpu_mem_mb()

        gpu_peak = gpu_after
        for i in range(min(1000, self.iterations)):
            retarget.infer(_make_body_pose(i * 0.02))
            if i % 100 == 0:
                g, _ = _gpu_mem_mb()
                gpu_peak = max(gpu_peak, g)
        g, _ = _gpu_mem_mb()
        gpu_peak = max(gpu_peak, g)

        return MemoryResult(
            rss_before_mb=rss_before, rss_after_mb=rss_after,
            gpu_before_mb=gpu_before, gpu_after_mb=gpu_after,
            gpu_peak_mb=gpu_peak, gpu_total_mb=gpu_total,
        )

    # ── 6. Stability / Jitter ────────────────────────────────

    def bench_stability(self, robot_type: str = "unitree_g1") -> Tuple[StabilityResult, List[float]]:
        self.cfg["robot"]["type"] = robot_type
        retarget = SonicRetarget(self.cfg)
        total = self.warmup + self.iterations
        poses = [_make_body_pose(t * 0.02) for t in range(total)]

        for i in range(self.warmup):
            retarget.infer(poses[i])

        times_ns: List[int] = []
        for i in range(self.iterations):
            t0 = time.perf_counter_ns()
            retarget.infer(poses[self.warmup + i])
            t1 = time.perf_counter_ns()
            times_ns.append(t1 - t0)

        ms = np.array(times_ns, dtype=np.float64) / 1e6
        med = float(np.median(ms))
        spikes = int(np.sum(ms > 2.0 * med))
        jitter = float(np.std(np.diff(ms)))

        return StabilityResult(
            total=self.iterations, spikes=spikes,
            spike_pct=100.0 * spikes / self.iterations,
            max_spike_ms=float(np.max(ms)),
            jitter_ms=jitter, median_ms=med,
        ), ms.tolist()

    # ── Run All ───────────────────────────────────────────────

    def run_all(self) -> BenchmarkResults:
        ts = datetime.now()
        r = self.results
        r.timestamp = ts.strftime("%Y-%m-%d %H:%M:%S")
        r.gpu_name = _gpu_name()
        r.python_version = platform.python_version()
        r.numpy_version = np.__version__
        r.ort_version = ort.__version__ if ort else "N/A"
        r.os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"

        model_dir = Path(self.cfg["sonic"]["model_dir"]).expanduser()
        has_onnx = (
            (model_dir / "model_encoder.onnx").exists()
            and (model_dir / "model_decoder.onnx").exists()
            and ort is not None
        )
        r.backend = "ONNX" if has_onnx else "Mock"
        robot = self.cfg["robot"]["type"]

        _print_banner(r.gpu_name, r.backend, robot)

        # 1
        _section("1. INFERENCE LATENCY", f"N={self.iterations}, warmup={self.warmup}")
        lat = self.bench_inference_latency(robot)
        r.latency = [{"stage": x.stage, **x.stats} for x in lat]
        _print_latency_table(lat)

        # 2
        _section("2. THROUGHPUT")
        thr = self.bench_throughput(robot)
        r.throughput = [{"label": x.label, "fps": x.fps, "batch_size": x.batch_size} for x in thr]
        _print_throughput_table(thr)

        # 3
        _section("3. FULL PIPELINE", "input → retarget → sim → update")
        pip = self.bench_full_pipeline(robot)
        r.pipeline = [{"stage": x.stage, "mean_ms": x.mean_ms} for x in pip]
        _print_pipeline_table(pip)

        # 4
        _section("4. PER-ROBOT COMPARISON")
        rob = self.bench_per_robot()
        r.per_robot = [{"robot": x.robot, "dof": x.dof, "mean_ms": x.mean_ms, "fps": x.fps} for x in rob]
        _print_robot_table(rob)

        # 5
        _section("5. STABILITY", f"N={self.iterations}")
        stab, trace = self.bench_stability(robot)
        r.stability = {
            "total": stab.total, "spikes": stab.spikes,
            "spike_pct": stab.spike_pct, "max_spike_ms": stab.max_spike_ms,
            "jitter_ms": stab.jitter_ms, "median_ms": stab.median_ms,
        }
        _print_stability_table(stab)

        # 6
        _section("6. MEMORY")
        mem = self.bench_memory(robot)
        r.memory = {
            "rss_before_mb": mem.rss_before_mb, "rss_after_mb": mem.rss_after_mb,
            "gpu_before_mb": mem.gpu_before_mb, "gpu_after_mb": mem.gpu_after_mb,
            "gpu_peak_mb": mem.gpu_peak_mb, "gpu_total_mb": mem.gpu_total_mb,
        }
        _print_memory_table(mem)

        # Save
        self._save(ts, trace)
        return r

    # ── Save results ──────────────────────────────────────────

    def _save(self, ts: datetime, trace: List[float]) -> None:
        out = _SCRIPT_DIR / "results"
        out.mkdir(parents=True, exist_ok=True)
        stamp = ts.strftime("%Y%m%d_%H%M%S")

        jp = out / f"sonic_benchmark_{stamp}.json"
        with open(jp, "w") as f:
            json.dump(self._as_dict(), f, indent=2, default=str)
        print(f"\n📄 JSON saved:     {jp}")

        mp = out / f"sonic_benchmark_{stamp}.md"
        with open(mp, "w") as f:
            f.write(self._markdown())
        print(f"📝 Markdown saved: {mp}")

        if HAS_MPL and trace:
            pp = out / f"sonic_latency_dist_{stamp}.png"
            self._plot(trace, pp)
            print(f"📊 Plot saved:     {pp}")

    def _as_dict(self) -> Dict[str, Any]:
        r = self.results
        return {
            "timestamp": r.timestamp,
            "system": {
                "gpu": r.gpu_name, "python": r.python_version,
                "numpy": r.numpy_version, "onnxruntime": r.ort_version,
                "os": r.os_info, "backend": r.backend,
            },
            "config": {
                "iterations": self.iterations, "warmup": self.warmup,
                "sustained_duration": self.sustained_duration,
                "target_hz": TARGET_HZ, "target_dt_ms": TARGET_DT_MS,
            },
            "latency": r.latency, "throughput": r.throughput,
            "pipeline": r.pipeline, "per_robot": r.per_robot,
            "stability": r.stability, "memory": r.memory,
        }

    def _markdown(self) -> str:
        r = self.results
        L = []
        L.append("# SONIC Retarget Inference Benchmark\n")
        L.append(f"**Date:** {r.timestamp}  ")
        L.append(f"**GPU:** {r.gpu_name}  ")
        L.append(f"**Backend:** {r.backend}  ")
        L.append(f"**Python:** {r.python_version} | **NumPy:** {r.numpy_version} | **ORT:** {r.ort_version}  ")
        L.append(f"**OS:** {r.os_info}  ")
        L.append(f"**Target:** {TARGET_HZ} Hz ({TARGET_DT_MS:.0f} ms budget)\n")

        L.append("## 1. Inference Latency\n")
        L.append("| Stage | Mean | Median | P95 | P99 | Max |")
        L.append("|-------|------|--------|-----|-----|-----|")
        for e in r.latency:
            L.append(f"| {e['stage']} | {_fmt_ms(e['mean_ms'])} | {_fmt_ms(e['median_ms'])} | {_fmt_ms(e['p95_ms'])} | {_fmt_ms(e['p99_ms'])} | {_fmt_ms(e['max_ms'])} |")

        L.append("\n## 2. Throughput\n")
        L.append("| Mode | FPS |")
        L.append("|------|-----|")
        for e in r.throughput:
            L.append(f"| {e['label']} | {_fmt_fps(e['fps'])} |")
        if r.throughput:
            h = r.throughput[0]["fps"] / TARGET_HZ
            L.append(f"\n**Headroom:** {h:,.0f}x {'✅' if h > 2 else '⚠️'}")

        L.append("\n## 3. Full Pipeline\n")
        L.append("| Stage | Mean |")
        L.append("|-------|------|")
        for e in r.pipeline:
            L.append(f"| {e['stage']} | {_fmt_ms(e['mean_ms'])} |")

        L.append("\n## 4. Per-Robot Comparison\n")
        L.append("| Robot | DOF | Mean (ms) | FPS |")
        L.append("|-------|-----|-----------|-----|")
        for e in r.per_robot:
            L.append(f"| {e['robot']} | {e['dof']} | {e['mean_ms']:.4f} | {_fmt_fps(e['fps'])} |")

        L.append("\n## 5. Stability\n")
        L.append("| Metric | Value |")
        L.append("|--------|-------|")
        L.append(f"| Spikes (>2x median) | {r.stability.get('spikes',0)} ({r.stability.get('spike_pct',0):.2f}%) |")
        L.append(f"| Max spike | {_fmt_ms(r.stability.get('max_spike_ms',0))} |")
        L.append(f"| Jitter | {_fmt_ms(r.stability.get('jitter_ms',0))} |")
        L.append(f"| Median | {_fmt_ms(r.stability.get('median_ms',0))} |")

        L.append("\n## 6. Memory\n")
        L.append("| Metric | Value |")
        L.append("|--------|-------|")
        L.append(f"| RSS before | {r.memory.get('rss_before_mb',0):.1f} MB |")
        L.append(f"| RSS after | {r.memory.get('rss_after_mb',0):.1f} MB |")
        L.append(f"| GPU before | {r.memory.get('gpu_before_mb',0):.0f} MB |")
        L.append(f"| GPU after model load | {r.memory.get('gpu_after_mb',0):.0f} MB |")
        L.append(f"| GPU peak | {r.memory.get('gpu_peak_mb',0):.0f} MB |")
        L.append(f"| GPU total | {r.memory.get('gpu_total_mb',0):.0f} MB |")

        return "\n".join(L) + "\n"

    def _plot(self, trace: List[float], path: Path) -> None:
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        ms = np.array(trace)
        med = np.median(ms)
        p99 = np.percentile(ms, 99)

        ax = axes[0]
        ax.hist(ms, bins=min(200, max(50, len(ms) // 50)), color="#2196F3", alpha=0.85, edgecolor="none")
        ax.axvline(med, color="green", ls="--", lw=1.5, label=f"Median: {_fmt_ms(med)}")
        ax.axvline(p99, color="red", ls="--", lw=1.5, label=f"P99: {_fmt_ms(p99)}")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"SONIC Inference Latency Distribution (N={len(ms):,})")
        ax.legend()

        ax2 = axes[1]
        ax2.plot(ms, lw=0.3, color="#2196F3", alpha=0.7)
        ax2.axhline(med, color="green", ls="--", lw=1, alpha=0.8)
        ax2.axhline(2 * med, color="red", ls="--", lw=1, alpha=0.8, label="2x median")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency Over Time")
        ax2.legend()

        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="SONIC Retarget Inference Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=str, default=None, help="Path to demo_config.yaml")
    parser.add_argument("--iterations", "-n", type=int, default=10000, help="Measurement iterations")
    parser.add_argument("--warmup", "-w", type=int, default=500, help="Warmup iterations")
    parser.add_argument("--sustained", type=float, default=10.0, help="Sustained throughput duration (s)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    bench = SonicBenchmark(
        config_path=args.config,
        iterations=args.iterations,
        warmup=args.warmup,
        sustained_duration=args.sustained,
    )
    bench.run_all()


if __name__ == "__main__":
    main()
