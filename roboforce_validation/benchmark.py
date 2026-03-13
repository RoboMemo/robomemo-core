#!/usr/bin/env python3
# Copyright (c) 2026, RoboForce Project. All rights reserved.
# SPDX-License-Identifier: Proprietary
"""Inference speed benchmark for the RoboForce screw-driving pipeline.

Benchmarks three stages of the pipeline:
    1. **Screw detection** — Object detection model inference
    2. **Skill policy** — GR00T N1.6 / π₀ VLA inference latency
    3. **End-to-end pipeline** — detect → plan → execute cycle

Measures mean, median, P95, P99, and max latency using ``perf_counter_ns``
with proper warmup. Inspired by the SONIC benchmark style.

Usage:
    python -m roboforce_validation.benchmark
    python -m roboforce_validation.benchmark --iterations 10000 --warmup 500
    python -m roboforce_validation.benchmark --policy openpi
    python -m roboforce_validation.benchmark --skip_detection
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_HZ = 50
TARGET_DT_MS = 1000.0 / TARGET_HZ  # 20 ms budget at 50 Hz

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "benchmarks"

# Input dimensions (matching RoboForce configuration)
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
IMAGE_CHANNELS = 3
STATE_DIM = 32  # 13 joints + 7 EE + 6 F/T + 6 misc
ACTION_DIM = 8  # delta_pose(6) + screw_rot + gripper


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_ms(ms: float) -> str:
    """Format milliseconds for display."""
    if ms < 0.001:
        return f"{ms * 1000:.2f}µs"
    if ms < 1.0:
        return f"{ms:.4f}ms"
    return f"{ms:.2f}ms"


def _fmt_fps(fps: float) -> str:
    return f"{fps:,.0f}"


def _rss_mb() -> float:
    """Get current RSS in MB."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            text=True, timeout=5,
        ).strip()
        return out
    except Exception:
        return "N/A"


def _gpu_mem() -> tuple[float, float]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,nounits,noheader", "--id=0"],
            text=True, timeout=5,
        ).strip()
        used, total = out.split(",")
        return float(used.strip()), float(total.strip())
    except Exception:
        return 0.0, 0.0


def _latency_stats(times_ns: list[int]) -> dict[str, float]:
    """Compute latency statistics from nanosecond timestamps."""
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


# Table drawing constants
W = 74


def _print_table_header(title: str) -> None:
    """Print a section header in box-drawing style."""
    print()
    print("┌" + "─" * W + "┐")
    print(f"│ {title:<{W-1}s}│")
    print("├" + "─" * 37 + "┬" + ("──────────┬" * 4) + "──────────┤")
    hdr = f"│ {'Stage':<36s}│{'Mean':>9s} │{'Median':>9s} │{'P95':>9s} │{'P99':>9s} │{'Max':>9s} │"
    print(hdr)
    print("├" + "─" * 37 + "┼" + ("──────────┼" * 4) + "──────────┤")


def _print_stats_row(label: str, stats: dict[str, float]) -> None:
    """Print one row of latency statistics."""
    print(
        f"│ {label:<36s}"
        f"│{_fmt_ms(stats['mean_ms']):>9s} "
        f"│{_fmt_ms(stats['median_ms']):>9s} "
        f"│{_fmt_ms(stats['p95_ms']):>9s} "
        f"│{_fmt_ms(stats['p99_ms']):>9s} "
        f"│{_fmt_ms(stats['max_ms']):>9s} │"
    )


def _print_table_footer() -> None:
    print("└" + "─" * 37 + "┴" + ("──────────┴" * 4) + "──────────┘")


# ---------------------------------------------------------------------------
# Mock Models (placeholders for real inference)
# ---------------------------------------------------------------------------

class MockScrewDetector:
    """Mock screw detection model.

    In production, wraps a YOLO / RT-DETR model for screw head detection.
    The mock performs equivalent numpy operations to simulate compute cost.
    """

    def __init__(self):
        self.loaded = False

    def load(self) -> float:
        """Load model (returns load time in seconds)."""
        t0 = time.perf_counter()
        # Simulate model loading — allocate representative weights
        self._weights = np.random.randn(256, 256).astype(np.float32)
        self._bias = np.random.randn(256).astype(np.float32)
        self.loaded = True
        return time.perf_counter() - t0

    def detect(self, image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Run detection on an image.

        Args:
            image: Input image ``(H, W, 3)``, uint8.

        Returns:
            Tuple of ``(bboxes, scores)`` where bboxes is ``(N, 4)`` and
            scores is ``(N,)``.
        """
        # Simulate detection compute (resize + feature extraction + NMS)
        small = image[::4, ::4, :].astype(np.float32) / 255.0
        features = small.reshape(small.shape[0], -1) @ self._weights[:small.shape[1], :64]
        scores = np.sigmoid(features.mean(axis=0)[:10])

        # Mock bounding boxes
        n_det = int(np.sum(scores > 0.5))
        bboxes = np.random.rand(max(n_det, 1), 4).astype(np.float32) * [640, 480, 640, 480]
        det_scores = np.sort(scores)[::-1][:max(n_det, 1)]
        return bboxes, det_scores


class MockPolicyModel:
    """Mock VLA policy model.

    Simulates the inference cost of GR00T N1.6 or OpenPI (π₀) policy models.
    """

    def __init__(self, policy_type: str = "gr00t"):
        self.policy_type = policy_type
        self.loaded = False

    def load(self) -> float:
        """Load model (returns load time in seconds)."""
        t0 = time.perf_counter()
        # Simulate model loading with representative tensor sizes
        if self.policy_type == "openpi":
            # π₀ has flow matching heads → more params
            self._image_encoder = np.random.randn(512, 512).astype(np.float32)
            self._flow_net = np.random.randn(512, ACTION_DIM * 16).astype(np.float32)
            self._num_denoise_steps = 10
        else:
            # GR00T — single-pass VLA
            self._image_encoder = np.random.randn(512, 512).astype(np.float32)
            self._action_head = np.random.randn(512, ACTION_DIM * 16).astype(np.float32)
            self._num_denoise_steps = 1

        self.loaded = True
        return time.perf_counter() - t0

    def predict(self, image: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Run policy inference.

        Args:
            image: Camera image ``(H, W, 3)``.
            state: Robot state ``(state_dim,)``.

        Returns:
            Action array ``(action_dim,)``.
        """
        # Image encoding (simulate ViT forward)
        patch = image[::8, ::8, :].astype(np.float32).reshape(-1, 3)
        tokens = patch[:512, :] @ self._image_encoder[:3, :512].T

        # State encoding
        state_emb = np.tile(state, (512 // STATE_DIM + 1))[:512]

        # Action decoding (GR00T = 1 step, π₀ = N denoising steps)
        combined = tokens.mean(axis=0) + state_emb
        for _ in range(self._num_denoise_steps):
            if self.policy_type == "openpi":
                action_logits = combined @ self._flow_net[:512, :ACTION_DIM]
            else:
                action_logits = combined @ self._action_head[:512, :ACTION_DIM]
            combined = combined * 0.99 + action_logits.mean() * 0.01  # fake residual

        action = np.tanh(action_logits).astype(np.float32)
        return action


class MockMotionPlanner:
    """Mock motion planner (trajectory interpolation)."""

    def plan(self, current_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Plan a trajectory from current state using the policy action.

        Args:
            current_state: Current robot state ``(state_dim,)``.
            action: Target action ``(action_dim,)``.

        Returns:
            Trajectory waypoints ``(horizon, action_dim)``.
        """
        horizon = 16
        # Linear interpolation (simplified)
        trajectory = np.linspace(
            current_state[:ACTION_DIM], action, horizon
        ).astype(np.float32)
        # Add smoothing
        for i in range(1, horizon - 1):
            trajectory[i] = 0.5 * trajectory[i] + 0.25 * (
                trajectory[i - 1] + trajectory[i + 1]
            )
        return trajectory


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

class RoboForceBenchmark:
    """Inference speed benchmark for the RoboForce pipeline."""

    def __init__(
        self,
        policy_type: str = "gr00t",
        iterations: int = 5000,
        warmup: int = 200,
        skip_detection: bool = False,
        sustained_duration: float = 5.0,
    ):
        self.policy_type = policy_type
        self.iterations = iterations
        self.warmup = warmup
        self.skip_detection = skip_detection
        self.sustained_duration = sustained_duration

        # Pin to single core for stable measurements
        try:
            os.sched_setaffinity(0, {0})
        except (AttributeError, OSError):
            pass

    def run(self) -> dict[str, Any]:
        """Run the full benchmark suite.

        Returns:
            Results dict with all latency measurements.
        """
        gpu_name = _gpu_name()
        ts = datetime.now()

        print()
        print("╔" + "═" * W + "╗")
        print("║" + "RoboForce Inference Benchmark".center(W) + "║")
        info = f"Policy: {self.policy_type} | GPU: {gpu_name}"
        print("║" + info.center(W) + "║")
        print("╠" + "═" * W + "╣")

        results: dict[str, Any] = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "gpu": gpu_name,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            },
            "config": {
                "policy_type": self.policy_type,
                "iterations": self.iterations,
                "warmup": self.warmup,
                "target_hz": TARGET_HZ,
                "image_size": [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS],
                "state_dim": STATE_DIM,
                "action_dim": ACTION_DIM,
            },
        }

        gc.collect()
        rss_before = _rss_mb()
        gpu_before, gpu_total = _gpu_mem()

        # ── Load models ──
        print(f"\n  Loading models...", end=" ", flush=True)
        detector = MockScrewDetector()
        policy = MockPolicyModel(self.policy_type)
        planner = MockMotionPlanner()

        det_load = detector.load()
        pol_load = policy.load()
        print(f"done (detector: {det_load:.3f}s, policy: {pol_load:.3f}s)")

        rss_after = _rss_mb()
        results["model_load"] = {
            "detector_s": round(det_load, 4),
            "policy_s": round(pol_load, 4),
            "rss_delta_mb": round(rss_after - rss_before, 1),
        }

        # Generate test inputs
        test_image = np.random.randint(
            0, 255, (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), dtype=np.uint8
        )
        test_state = np.random.randn(STATE_DIM).astype(np.float32) * 0.1

        total = self.warmup + self.iterations

        # ══════════════════════════════════════════════════════════════
        # 1. INFERENCE LATENCY
        # ══════════════════════════════════════════════════════════════
        _print_table_header(
            f"1. INFERENCE LATENCY (N={self.iterations}, warmup={self.warmup})"
        )

        # ── Screw detection ──
        if not self.skip_detection:
            det_times: list[int] = []
            for i in range(total):
                t0 = time.perf_counter_ns()
                detector.detect(test_image)
                t1 = time.perf_counter_ns()
                if i >= self.warmup:
                    det_times.append(t1 - t0)
            det_stats = _latency_stats(det_times)
            results["detection_latency"] = det_stats
            _print_stats_row("Screw detection (YOLO/RT-DETR)", det_stats)
        else:
            det_stats = {"mean_ms": 0.0, "p99_ms": 0.0}

        # ── Policy inference ──
        pol_times: list[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            policy.predict(test_image, test_state)
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                pol_times.append(t1 - t0)
        pol_stats = _latency_stats(pol_times)
        results["policy_latency"] = pol_stats
        policy_label = f"Policy ({self.policy_type})"
        _print_stats_row(policy_label, pol_stats)

        # ── Motion planning ──
        test_action = np.random.randn(ACTION_DIM).astype(np.float32) * 0.1
        plan_times: list[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            planner.plan(test_state, test_action)
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                plan_times.append(t1 - t0)
        plan_stats = _latency_stats(plan_times)
        results["planning_latency"] = plan_stats
        _print_stats_row("Motion planning (interpolation)", plan_stats)

        # ── End-to-end pipeline ──
        e2e_times: list[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            if not self.skip_detection:
                bboxes, scores = detector.detect(test_image)
            action = policy.predict(test_image, test_state)
            trajectory = planner.plan(test_state, action)
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                e2e_times.append(t1 - t0)
        e2e_stats = _latency_stats(e2e_times)
        results["e2e_latency"] = e2e_stats
        _print_stats_row("End-to-end (detect→plan→execute)", e2e_stats)

        # Summary row
        print("├" + "─" * 37 + "┼" + ("──────────┼" * 4) + "──────────┤")
        headroom = TARGET_DT_MS / e2e_stats["mean_ms"] if e2e_stats["mean_ms"] > 0 else float("inf")
        icon = "✅" if headroom > 2 else ("⚠️" if headroom > 1 else "❌")
        summary = (
            f"Pipeline: {_fmt_ms(e2e_stats['mean_ms'])} mean, "
            f"{_fmt_ms(e2e_stats['p99_ms'])} P99 → {headroom:.0f}x headroom {icon}"
        )
        print(f"│ {summary:<{W-1}s}│")
        _print_table_footer()
        results["headroom_vs_50hz"] = headroom

        # ══════════════════════════════════════════════════════════════
        # 2. THROUGHPUT
        # ══════════════════════════════════════════════════════════════
        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 2. THROUGHPUT".ljust(W + 1) + "│")
        print("├" + "─" * 42 + "┬" + "─" * 31 + "┤")

        throughput_results = []

        # Sequential E2E
        for _ in range(self.warmup):
            if not self.skip_detection:
                detector.detect(test_image)
            policy.predict(test_image, test_state)
            planner.plan(test_state, test_action)

        t0 = time.perf_counter_ns()
        for _ in range(self.iterations):
            if not self.skip_detection:
                detector.detect(test_image)
            policy.predict(test_image, test_state)
            planner.plan(test_state, test_action)
        ns = time.perf_counter_ns() - t0
        seq_fps = self.iterations / (ns / 1e9)
        print(f"│ {'Sequential E2E FPS':<41s}│ {_fmt_fps(seq_fps) + ' fps':>30s} │")
        throughput_results.append({"label": "Sequential E2E", "fps": seq_fps})

        # Policy only throughput
        t0 = time.perf_counter_ns()
        for _ in range(self.iterations):
            policy.predict(test_image, test_state)
        ns = time.perf_counter_ns() - t0
        pol_fps = self.iterations / (ns / 1e9)
        print(f"│ {'Policy only FPS':<41s}│ {_fmt_fps(pol_fps) + ' fps':>30s} │")
        throughput_results.append({"label": "Policy only", "fps": pol_fps})

        # Sustained E2E
        count = 0
        t_s = time.perf_counter()
        t_e = t_s + self.sustained_duration
        while time.perf_counter() < t_e:
            if not self.skip_detection:
                detector.detect(test_image)
            policy.predict(test_image, test_state)
            planner.plan(test_state, test_action)
            count += 1
        dur = time.perf_counter() - t_s
        sus_fps = count / dur
        label = f"Sustained E2E ({self.sustained_duration:.0f}s)"
        print(f"│ {label:<41s}│ {_fmt_fps(sus_fps) + ' fps':>30s} │")
        throughput_results.append({"label": label, "fps": sus_fps})

        headroom_t = seq_fps / TARGET_HZ
        icon_t = "✅" if headroom_t > 10 else ("⚠️" if headroom_t > 2 else "❌")
        print("├" + "─" * 42 + "┼" + "─" * 31 + "┤")
        print(f"│ {'Target: ' + str(TARGET_HZ) + 'Hz real-time':<41s}│ {icon_t + ' ' + str(int(headroom_t)) + 'x headroom':>30s} │")
        print("└" + "─" * 42 + "┴" + "─" * 31 + "┘")
        results["throughput"] = throughput_results

        # ══════════════════════════════════════════════════════════════
        # 3. STABILITY / JITTER
        # ══════════════════════════════════════════════════════════════
        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 3. STABILITY / JITTER (N={self.iterations})".ljust(W + 1) + "│")
        print("├" + "─" * 42 + "┬" + "─" * 31 + "┤")

        ms_arr = np.array(e2e_times, dtype=np.float64) / 1e6
        med = float(np.median(ms_arr))
        spikes = int(np.sum(ms_arr > 2 * med))
        jitter = float(np.std(np.diff(ms_arr)))
        max_val = float(np.max(ms_arr))

        rows = [
            ("Median E2E latency", _fmt_ms(med)),
            ("Latency spikes (>2x median)", f"{spikes} ({100 * spikes / len(ms_arr):.2f}%)"),
            ("Max latency", _fmt_ms(max_val)),
            ("Jitter (std of Δlatency)", _fmt_ms(jitter)),
            (f"Within {TARGET_DT_MS:.0f}ms budget (50Hz)",
             f"{'✅ YES' if max_val < TARGET_DT_MS else '❌ NO (max=' + _fmt_ms(max_val) + ')'}"),
        ]
        for label, val in rows:
            print(f"│ {label:<41s}│ {val:>30s} │")
        print("└" + "─" * 42 + "┴" + "─" * 31 + "┘")
        results["stability"] = {
            "median_ms": med,
            "spikes": spikes,
            "spike_pct": 100 * spikes / len(ms_arr),
            "max_ms": max_val,
            "jitter_ms": jitter,
        }

        # ══════════════════════════════════════════════════════════════
        # 4. POLICY COMPARISON (GR00T vs OpenPI)
        # ══════════════════════════════════════════════════════════════
        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 4. POLICY COMPARISON (GR00T vs OpenPI)".ljust(W + 1) + "│")
        print("├" + "─" * 30 + "┬" + "─" * 21 + "┬" + "─" * 21 + "┤")
        print(f"│ {'Metric':<29s}│{'GR00T N1.6':>20s} │{'OpenPI (π₀)':>20s} │")
        print("├" + "─" * 30 + "┼" + "─" * 21 + "┼" + "─" * 21 + "┤")

        comparison = {}
        for ptype in ["gr00t", "openpi"]:
            p = MockPolicyModel(ptype)
            p.load()
            times: list[int] = []
            for i in range(self.warmup + min(2000, self.iterations)):
                t0 = time.perf_counter_ns()
                p.predict(test_image, test_state)
                t1 = time.perf_counter_ns()
                if i >= self.warmup:
                    times.append(t1 - t0)
            comparison[ptype] = _latency_stats(times)

        g = comparison["gr00t"]
        o = comparison["openpi"]
        comp_rows = [
            ("Mean latency", _fmt_ms(g["mean_ms"]), _fmt_ms(o["mean_ms"])),
            ("P99 latency", _fmt_ms(g["p99_ms"]), _fmt_ms(o["p99_ms"])),
            ("Max latency", _fmt_ms(g["max_ms"]), _fmt_ms(o["max_ms"])),
            ("FPS (1/mean)", _fmt_fps(1000 / g["mean_ms"]) if g["mean_ms"] > 0 else "N/A",
             _fmt_fps(1000 / o["mean_ms"]) if o["mean_ms"] > 0 else "N/A"),
        ]
        for label, gv, ov in comp_rows:
            print(f"│ {label:<29s}│{gv:>20s} │{ov:>20s} │")

        speedup = o["mean_ms"] / g["mean_ms"] if g["mean_ms"] > 0 else 0
        print("├" + "─" * 30 + "┼" + "─" * 21 + "┼" + "─" * 21 + "┤")
        icon_s = "🚀" if speedup > 1.5 else "→"
        print(f"│ {'GR00T speedup vs π₀':<29s}│ {icon_s + f' {speedup:.1f}x faster':>19s} │{'':>20s} │")
        print("└" + "─" * 30 + "┴" + "─" * 21 + "┴" + "─" * 21 + "┘")
        results["policy_comparison"] = {
            "gr00t": comparison["gr00t"],
            "openpi": comparison["openpi"],
            "gr00t_speedup": speedup,
        }

        # ══════════════════════════════════════════════════════════════
        # 5. MEMORY
        # ══════════════════════════════════════════════════════════════
        gpu_after, _ = _gpu_mem()
        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 5. MEMORY".ljust(W + 1) + "│")
        print("├" + "─" * 42 + "┬" + "─" * 31 + "┤")
        mem_rows = [
            ("RSS before → after load", f"{rss_before:.1f} MB → {rss_after:.1f} MB (+{rss_after - rss_before:.1f})"),
            ("GPU before → after", f"{gpu_before:.0f} MB → {gpu_after:.0f} MB (+{gpu_after - gpu_before:.0f})"),
            ("GPU total", f"{gpu_total:.0f} MB"),
        ]
        for label, val in mem_rows:
            print(f"│ {label:<41s}│ {val:>30s} │")
        print("└" + "─" * 42 + "┴" + "─" * 31 + "┘")
        results["memory"] = {
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "gpu_before_mb": gpu_before,
            "gpu_after_mb": gpu_after,
            "gpu_total_mb": gpu_total,
        }

        # ══════════════════════════════════════════════════════════════
        # DEPLOYMENT READINESS
        # ══════════════════════════════════════════════════════════════
        print()
        print("╔" + "═" * W + "╗")
        print("║" + "DEPLOYMENT READINESS SUMMARY".center(W) + "║")
        print("╠" + "═" * W + "╣")

        checks = [
            (f"E2E latency < 20ms (50Hz budget)",
             e2e_stats["p99_ms"] < TARGET_DT_MS,
             f"P99={_fmt_ms(e2e_stats['p99_ms'])}"),
            (f"E2E latency < 10ms (100Hz budget)",
             e2e_stats["p99_ms"] < 10.0,
             f"P99={_fmt_ms(e2e_stats['p99_ms'])}"),
            (f"No spikes > 2x median",
             spikes == 0,
             f"{spikes} spikes"),
            (f"Jitter < 1ms",
             jitter < 1.0,
             _fmt_ms(jitter)),
            (f"Policy FPS > {TARGET_HZ}Hz",
             pol_fps > TARGET_HZ,
             f"{_fmt_fps(pol_fps)} fps"),
        ]
        for label, ok, detail in checks:
            icon = "✅" if ok else "❌"
            print(f"║  {icon} {label:<40s} {detail:>{W - 46}s}  ║")
        print("╚" + "═" * W + "╝")
        results["deployment_checks"] = [
            {"check": c[0], "pass": c[1], "detail": c[2]} for c in checks
        ]

        # Save results
        self._save(ts, results)
        return results

    def _save(self, ts: datetime, results: dict[str, Any]) -> None:
        """Save results to JSON and Markdown."""
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = ts.strftime("%Y%m%d_%H%M%S")

        # JSON
        jp = RESULTS_DIR / f"roboforce_benchmark_{self.policy_type}_{stamp}.json"
        with open(jp, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 JSON saved: {jp}")

        # Markdown
        mp = RESULTS_DIR / f"roboforce_benchmark_{self.policy_type}_{stamp}.md"
        with open(mp, "w") as f:
            f.write(self._markdown(results))
        print(f"📝 Markdown saved: {mp}")

    def _markdown(self, r: dict[str, Any]) -> str:
        """Render the benchmark results as Markdown."""
        lines = [
            "# RoboForce Inference Benchmark\n",
            f"**Date:** {r['timestamp']}  ",
            f"**GPU:** {r['system']['gpu']}  ",
            f"**Policy:** {r['config']['policy_type']}  ",
            f"**Iterations:** {r['config']['iterations']} (warmup: {r['config']['warmup']})\n",
            "## Key Results\n",
            "| Metric | Value |",
            "|--------|-------|",
        ]

        if "detection_latency" in r:
            lines.append(
                f"| Detection latency (mean) | {_fmt_ms(r['detection_latency']['mean_ms'])} |"
            )
        lines.extend([
            f"| Policy latency (mean) | {_fmt_ms(r['policy_latency']['mean_ms'])} |",
            f"| E2E latency (mean) | {_fmt_ms(r['e2e_latency']['mean_ms'])} |",
            f"| E2E latency (P99) | {_fmt_ms(r['e2e_latency']['p99_ms'])} |",
            f"| Headroom vs {TARGET_HZ}Hz | {r.get('headroom_vs_50hz', 0):.0f}x |",
            f"| Sequential FPS | {_fmt_fps(r['throughput'][0]['fps'])} |",
        ])

        if "policy_comparison" in r:
            lines.extend([
                "",
                "## GR00T vs OpenPI\n",
                "| Metric | GR00T | OpenPI |",
                "|--------|-------|--------|",
                f"| Mean latency | {_fmt_ms(r['policy_comparison']['gr00t']['mean_ms'])} "
                f"| {_fmt_ms(r['policy_comparison']['openpi']['mean_ms'])} |",
                f"| P99 latency | {_fmt_ms(r['policy_comparison']['gr00t']['p99_ms'])} "
                f"| {_fmt_ms(r['policy_comparison']['openpi']['p99_ms'])} |",
                f"| GR00T speedup | {r['policy_comparison']['gr00t_speedup']:.1f}x | |",
            ])

        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoboForce — Inference Speed Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--policy", type=str, default="gr00t",
        choices=["gr00t", "openpi"],
        help="Primary policy to benchmark",
    )
    parser.add_argument(
        "--iterations", "-n", type=int, default=5000,
        help="Measurement iterations",
    )
    parser.add_argument(
        "--warmup", "-w", type=int, default=200,
        help="Warmup iterations",
    )
    parser.add_argument(
        "--skip_detection", action="store_true",
        help="Skip screw detection benchmark",
    )
    parser.add_argument(
        "--sustained", type=float, default=5.0,
        help="Sustained throughput duration (seconds)",
    )

    args = parser.parse_args()

    bench = RoboForceBenchmark(
        policy_type=args.policy,
        iterations=args.iterations,
        warmup=args.warmup,
        skip_detection=args.skip_detection,
        sustained_duration=args.sustained,
    )
    bench.run()


if __name__ == "__main__":
    main()
