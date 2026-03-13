#!/usr/bin/env python3
"""
SONIC Real ONNX Model Inference Benchmark.

Directly benchmarks the actual SONIC encoder + decoder ONNX models
with correct input dimensions from observation_config.yaml.

Model specs (nvidia/GEAR-SONIC):
  Encoder: obs_dict [1, 1762] → encoded_tokens [1, 64]
  Decoder: obs_dict [1, 994]  → action [1, 29]

Usage:
    python3 cross_embodiment_retarget_demo/benchmarks/sonic_onnx_benchmark.py
    python3 cross_embodiment_retarget_demo/benchmarks/sonic_onnx_benchmark.py --iterations 10000 --warmup 500
    python3 cross_embodiment_retarget_demo/benchmarks/sonic_onnx_benchmark.py --device cpu
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENCODER_INPUT_DIM = 1762
ENCODER_OUTPUT_DIM = 64
DECODER_INPUT_DIM = 994
DECODER_OUTPUT_DIM = 29
TARGET_HZ = 50
TARGET_DT_MS = 1000.0 / TARGET_HZ  # 20 ms

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints" / "sonic"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ---------------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------------
try:
    import onnxruntime as ort
except ImportError:
    print("ERROR: onnxruntime not installed. Run: pip install onnxruntime-gpu")
    sys.exit(1)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


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
    return f"{fps:,.0f}"


def _rss_mb() -> float:
    if HAS_PSUTIL:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _gpu_mem() -> Tuple[float, float]:
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


def _gpu_name() -> str:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "--id=0"],
            text=True, timeout=5,
        ).strip()
        return out
    except Exception:
        return "N/A"


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


def _print_stats_row(label: str, stats: Dict[str, float]) -> None:
    print(
        f"│ {label:<36s}"
        f"│{_fmt_ms(stats['mean_ms']):>9s} "
        f"│{_fmt_ms(stats['median_ms']):>9s} "
        f"│{_fmt_ms(stats['p95_ms']):>9s} "
        f"│{_fmt_ms(stats['p99_ms']):>9s} "
        f"│{_fmt_ms(stats['max_ms']):>9s} │"
    )


W = 74


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
class SonicOnnxBenchmark:
    """Benchmark real SONIC ONNX encoder + decoder models."""

    def __init__(
        self,
        device: str = "cuda",
        iterations: int = 10000,
        warmup: int = 500,
        sustained_duration: float = 10.0,
    ):
        self.device = device
        self.iterations = iterations
        self.warmup = warmup
        self.sustained_duration = sustained_duration

        # Validate models exist
        self.encoder_path = CHECKPOINT_DIR / "model_encoder.onnx"
        self.decoder_path = CHECKPOINT_DIR / "model_decoder.onnx"
        if not self.encoder_path.exists():
            raise FileNotFoundError(f"Encoder not found: {self.encoder_path}")
        if not self.decoder_path.exists():
            raise FileNotFoundError(f"Decoder not found: {self.decoder_path}")

        # Select providers
        if device == "cuda" and "CUDAExecutionProvider" in ort.get_available_providers():
            self.providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            self.device_label = "CUDA (GPU)"
        else:
            self.providers = ["CPUExecutionProvider"]
            self.device_label = "CPU"
            if device == "cuda":
                print("⚠️  CUDA not available, falling back to CPU")

        # Pin to single core for stable measurements
        try:
            os.sched_setaffinity(0, {0})
        except (AttributeError, OSError):
            pass

    def _load_sessions(self) -> Tuple[ort.InferenceSession, ort.InferenceSession]:
        """Load ONNX sessions with optimizations."""
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 1  # Single thread for deterministic timing
        opts.inter_op_num_threads = 1

        enc = ort.InferenceSession(
            str(self.encoder_path), sess_options=opts, providers=self.providers
        )
        dec = ort.InferenceSession(
            str(self.decoder_path), sess_options=opts, providers=self.providers
        )
        return enc, dec

    def _random_encoder_input(self, batch: int = 1) -> np.ndarray:
        """Generate realistic random encoder input [batch, 1762]."""
        return np.random.randn(batch, ENCODER_INPUT_DIM).astype(np.float32) * 0.1

    def _random_decoder_input(self, batch: int = 1) -> np.ndarray:
        """Generate realistic random decoder input [batch, 994]."""
        return np.random.randn(batch, DECODER_INPUT_DIM).astype(np.float32) * 0.1

    def run(self) -> Dict[str, Any]:
        gpu_name = _gpu_name()
        ts = datetime.now()
        print()
        print("╔" + "═" * W + "╗")
        print("║" + "SONIC ONNX Model Inference Benchmark".center(W) + "║")
        info = f"GPU: {gpu_name} | Device: {self.device_label}"
        print("║" + info.center(W) + "║")
        print("║" + f"Encoder: {self.encoder_path.stat().st_size / 1e6:.1f}MB | Decoder: {self.decoder_path.stat().st_size / 1e6:.1f}MB".center(W) + "║")
        print("╠" + "═" * W + "╣")

        results: Dict[str, Any] = {
            "timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
            "system": {
                "gpu": gpu_name,
                "device": self.device_label,
                "python": platform.python_version(),
                "numpy": np.__version__,
                "onnxruntime": ort.__version__,
                "providers": self.providers,
                "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            },
            "config": {
                "iterations": self.iterations,
                "warmup": self.warmup,
                "encoder_input_dim": ENCODER_INPUT_DIM,
                "encoder_output_dim": ENCODER_OUTPUT_DIM,
                "decoder_input_dim": DECODER_INPUT_DIM,
                "decoder_output_dim": DECODER_OUTPUT_DIM,
            },
        }

        # ── 0. Memory: before load ──
        gc.collect()
        rss_before = _rss_mb()
        gpu_before, gpu_total = _gpu_mem()

        # ── Load models ──
        print(f"\n  Loading models ({self.device_label})…", end=" ", flush=True)
        t0 = time.perf_counter()
        enc_sess, dec_sess = self._load_sessions()
        load_time = time.perf_counter() - t0
        print(f"done in {load_time:.2f}s")

        rss_after = _rss_mb()
        gpu_after, _ = _gpu_mem()

        results["model_load_time_s"] = load_time
        results["memory"] = {
            "rss_before_mb": rss_before,
            "rss_after_mb": rss_after,
            "rss_delta_mb": rss_after - rss_before,
            "gpu_before_mb": gpu_before,
            "gpu_after_mb": gpu_after,
            "gpu_delta_mb": gpu_after - gpu_before,
            "gpu_total_mb": gpu_total,
        }

        # Verify model I/O
        enc_in = enc_sess.get_inputs()[0]
        enc_out = enc_sess.get_outputs()[0]
        dec_in = dec_sess.get_inputs()[0]
        dec_out = dec_sess.get_outputs()[0]
        print(f"  Encoder: {enc_in.name} {enc_in.shape} → {enc_out.name} {enc_out.shape}")
        print(f"  Decoder: {dec_in.name} {dec_in.shape} → {dec_out.name} {dec_out.shape}")

        # ══════════════════════════════════════════════════════════════
        # 1. INFERENCE LATENCY
        # ══════════════════════════════════════════════════════════════
        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 1. INFERENCE LATENCY (N={self.iterations}, warmup={self.warmup})".ljust(W + 1) + "│")
        print("├" + "─" * 37 + "┬" + ("──────────┬" * 4) + "──────────┤")
        hdr = f"│ {'Stage':<36s}│{'Mean':>9s} │{'Median':>9s} │{'P95':>9s} │{'P99':>9s} │{'Max':>9s} │"
        print(hdr)
        print("├" + "─" * 37 + "┼" + ("──────────┼" * 4) + "──────────┤")

        total = self.warmup + self.iterations

        # ── Encoder only ──
        enc_input = self._random_encoder_input()
        enc_times: List[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            enc_sess.run(None, {enc_in.name: enc_input})
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                enc_times.append(t1 - t0)
        enc_stats = _latency_stats(enc_times)
        results["encoder_latency"] = enc_stats
        _print_stats_row("Encoder (obs→tokens)", enc_stats)

        # ── Decoder only ──
        dec_input = self._random_decoder_input()
        dec_times: List[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            dec_sess.run(None, {dec_in.name: dec_input})
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                dec_times.append(t1 - t0)
        dec_stats = _latency_stats(dec_times)
        results["decoder_latency"] = dec_stats
        _print_stats_row("Decoder (obs+tokens→action)", dec_stats)

        # ── End-to-end: encoder + decoder ──
        enc_input_e2e = self._random_encoder_input()
        e2e_times: List[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            tokens = enc_sess.run(None, {enc_in.name: enc_input_e2e})[0]
            # In real pipeline, tokens (64-dim) are embedded into decoder's 994-dim input
            # We simulate this with pre-generated decoder input
            dec_sess.run(None, {dec_in.name: dec_input})
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                e2e_times.append(t1 - t0)
        e2e_stats = _latency_stats(e2e_times)
        results["e2e_latency"] = e2e_stats
        _print_stats_row("End-to-end (encoder+decoder)", e2e_stats)

        # ── Post-processing (numpy ops) ──
        prev_targets = np.zeros(DECODER_OUTPUT_DIM, dtype=np.float32)
        raw = np.random.randn(DECODER_OUTPUT_DIM).astype(np.float32) * 0.1
        pp_times: List[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            max_d = 0.8 * 10.0 * 0.02
            delta = np.clip(raw - prev_targets, -max_d, max_d)
            tgt = prev_targets + delta
            tgt = np.clip(tgt, -3.14, 3.14)
            prev_targets = tgt
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                pp_times.append(t1 - t0)
        pp_stats = _latency_stats(pp_times)
        results["postproc_latency"] = pp_stats
        _print_stats_row("Post-processing (vel limit+clamp)", pp_stats)

        # ── Total pipeline estimate ──
        total_mean = e2e_stats["mean_ms"] + pp_stats["mean_ms"]
        total_p99 = e2e_stats["p99_ms"] + pp_stats["p99_ms"]
        print("├" + "─" * 37 + "┼" + ("──────────┼" * 4) + "──────────┤")
        headroom = TARGET_DT_MS / total_mean if total_mean > 0 else float("inf")
        icon = "✅" if headroom > 2 else ("⚠️" if headroom > 1 else "❌")
        summary = f"Total pipeline: {_fmt_ms(total_mean)} mean, {_fmt_ms(total_p99)} P99 → {headroom:.0f}x headroom {icon}"
        print(f"│ {summary:<{W-1}s}│")
        print("└" + "─" * W + "┘")
        results["total_pipeline_mean_ms"] = total_mean
        results["total_pipeline_p99_ms"] = total_p99
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
            enc_sess.run(None, {enc_in.name: enc_input})
            dec_sess.run(None, {dec_in.name: dec_input})
        t0 = time.perf_counter_ns()
        for _ in range(self.iterations):
            enc_sess.run(None, {enc_in.name: enc_input})
            dec_sess.run(None, {dec_in.name: dec_input})
        ns = time.perf_counter_ns() - t0
        seq_fps = self.iterations / (ns / 1e9)
        print(f"│ {'Sequential E2E FPS':<41s}│ {_fmt_fps(seq_fps) + ' fps':>30s} │")
        throughput_results.append({"label": "Sequential E2E", "fps": seq_fps})

        # Encoder only throughput
        t0 = time.perf_counter_ns()
        for _ in range(self.iterations):
            enc_sess.run(None, {enc_in.name: enc_input})
        ns = time.perf_counter_ns() - t0
        enc_fps = self.iterations / (ns / 1e9)
        print(f"│ {'Encoder only FPS':<41s}│ {_fmt_fps(enc_fps) + ' fps':>30s} │")
        throughput_results.append({"label": "Encoder only", "fps": enc_fps})

        # Decoder only throughput
        t0 = time.perf_counter_ns()
        for _ in range(self.iterations):
            dec_sess.run(None, {dec_in.name: dec_input})
        ns = time.perf_counter_ns() - t0
        dec_fps = self.iterations / (ns / 1e9)
        print(f"│ {'Decoder only FPS':<41s}│ {_fmt_fps(dec_fps) + ' fps':>30s} │")
        throughput_results.append({"label": "Decoder only", "fps": dec_fps})

        # Sustained E2E
        count = 0
        t_s = time.perf_counter()
        t_e = t_s + self.sustained_duration
        while time.perf_counter() < t_e:
            enc_sess.run(None, {enc_in.name: enc_input})
            dec_sess.run(None, {dec_in.name: dec_input})
            count += 1
        dur = time.perf_counter() - t_s
        sus_fps = count / dur
        print(f"│ {'Sustained E2E FPS (' + str(int(self.sustained_duration)) + 's)':<41s}│ {_fmt_fps(sus_fps) + ' fps':>30s} │")
        throughput_results.append({"label": f"Sustained E2E ({self.sustained_duration:.0f}s)", "fps": sus_fps})

        headroom_t = seq_fps / TARGET_HZ
        icon_t = "✅" if headroom_t > 10 else ("⚠️" if headroom_t > 2 else "❌")
        print("├" + "─" * 42 + "┼" + "─" * 31 + "┤")
        print(f"│ {'Target: ' + str(TARGET_HZ) + 'Hz real-time':<41s}│ {icon_t + ' ' + str(int(headroom_t)) + 'x headroom':>30s} │")
        print("└" + "─" * 42 + "┴" + "─" * 31 + "┘")
        results["throughput"] = throughput_results

        # ══════════════════════════════════════════════════════════════
        # 3. BATCH SIZE SCALING
        # ══════════════════════════════════════════════════════════════
        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 3. BATCH SIZE SCALING (encoder)".ljust(W + 1) + "│")
        print("├" + "─" * 20 + "┬" + "─" * 15 + "┬" + "─" * 15 + "┬" + "─" * 22 + "┤")
        print(f"│ {'Batch Size':<19s}│{'Latency':>14s} │{'Throughput':>14s} │{'Per-item':>21s} │")
        print("├" + "─" * 20 + "┼" + "─" * 15 + "┼" + "─" * 15 + "┼" + "─" * 22 + "┤")

        batch_results = []
        for bs in [1, 2, 4, 8, 16, 32]:
            enc_batch = np.random.randn(bs, ENCODER_INPUT_DIM).astype(np.float32) * 0.1
            # Warmup
            for _ in range(100):
                try:
                    enc_sess.run(None, {enc_in.name: enc_batch})
                except Exception:
                    break
            # Measure
            times_b: List[int] = []
            try:
                for _ in range(min(2000, self.iterations)):
                    t0 = time.perf_counter_ns()
                    enc_sess.run(None, {enc_in.name: enc_batch})
                    t1 = time.perf_counter_ns()
                    times_b.append(t1 - t0)
            except Exception as e:
                print(f"│ {bs:<19d}│ {'FAILED':>14s} │ {str(e)[:14]:>14s} │ {'N/A':>21s} │")
                batch_results.append({"batch_size": bs, "error": str(e)})
                continue

            bstats = _latency_stats(times_b)
            ips = bs * 1000.0 / bstats["mean_ms"] if bstats["mean_ms"] > 0 else 0
            per_item = bstats["mean_ms"] / bs
            print(
                f"│ {bs:<19d}"
                f"│ {_fmt_ms(bstats['mean_ms']):>13s} "
                f"│ {_fmt_fps(ips) + '/s':>13s} "
                f"│ {_fmt_ms(per_item) + '/item':>20s} │"
            )
            batch_results.append({
                "batch_size": bs,
                "mean_ms": bstats["mean_ms"],
                "throughput_ips": ips,
                "per_item_ms": per_item,
            })
        print("└" + "─" * 20 + "┴" + "─" * 15 + "┴" + "─" * 15 + "┴" + "─" * 22 + "┘")
        results["batch_scaling"] = batch_results

        # ══════════════════════════════════════════════════════════════
        # 4. CUDA vs CPU COMPARISON
        # ══════════════════════════════════════════════════════════════
        if "CUDAExecutionProvider" in ort.get_available_providers():
            print()
            print("┌" + "─" * W + "┐")
            print(f"│ 4. CUDA vs CPU COMPARISON".ljust(W + 1) + "│")
            print("├" + "─" * 30 + "┬" + "─" * 21 + "┬" + "─" * 21 + "┤")
            print(f"│ {'Stage':<29s}│{'CUDA':>20s} │{'CPU':>20s} │")
            print("├" + "─" * 30 + "┼" + "─" * 21 + "┼" + "─" * 21 + "┤")

            comparison = []
            for label, model_path, input_data, input_name in [
                ("Encoder", str(self.encoder_path), enc_input, enc_in.name),
                ("Decoder", str(self.decoder_path), dec_input, dec_in.name),
            ]:
                row = {"stage": label}
                for prov_label, provs in [
                    ("CUDA", ["CUDAExecutionProvider", "CPUExecutionProvider"]),
                    ("CPU", ["CPUExecutionProvider"]),
                ]:
                    opts_c = ort.SessionOptions()
                    opts_c.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                    opts_c.intra_op_num_threads = 1
                    sess = ort.InferenceSession(model_path, sess_options=opts_c, providers=provs)
                    in_name = sess.get_inputs()[0].name
                    # Warmup
                    for _ in range(200):
                        sess.run(None, {in_name: input_data})
                    times_c: List[int] = []
                    for _ in range(min(5000, self.iterations)):
                        t0 = time.perf_counter_ns()
                        sess.run(None, {in_name: input_data})
                        t1 = time.perf_counter_ns()
                        times_c.append(t1 - t0)
                    s = _latency_stats(times_c)
                    row[prov_label.lower() + "_mean_ms"] = s["mean_ms"]
                    row[prov_label.lower() + "_p99_ms"] = s["p99_ms"]

                cuda_str = f"{_fmt_ms(row['cuda_mean_ms'])} (P99: {_fmt_ms(row['cuda_p99_ms'])})"
                cpu_str = f"{_fmt_ms(row['cpu_mean_ms'])} (P99: {_fmt_ms(row['cpu_p99_ms'])})"
                speedup = row["cpu_mean_ms"] / row["cuda_mean_ms"] if row["cuda_mean_ms"] > 0 else 0
                row["speedup"] = speedup
                print(f"│ {label:<29s}│{cuda_str:>20s} │{cpu_str:>20s} │")
                comparison.append(row)

            print("├" + "─" * 30 + "┼" + "─" * 21 + "┼" + "─" * 21 + "┤")
            for r in comparison:
                sp = r.get("speedup", 0)
                icon_s = "🚀" if sp > 2 else ("→" if sp > 0.8 else "🐢")
                print(f"│ {r['stage'] + ' speedup':<29s}│ {icon_s + f' {sp:.1f}x':>19s} │{'':>20s} │")
            print("└" + "─" * 30 + "┴" + "─" * 21 + "┴" + "─" * 21 + "┘")
            results["cuda_vs_cpu"] = comparison

        # ══════════════════════════════════════════════════════════════
        # 5. STABILITY / JITTER (E2E)
        # ══════════════════════════════════════════════════════════════
        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 5. STABILITY / JITTER (N={self.iterations})".ljust(W + 1) + "│")
        print("├" + "─" * 42 + "┬" + "─" * 31 + "┤")

        stab_times: List[int] = []
        for i in range(total):
            t0 = time.perf_counter_ns()
            enc_sess.run(None, {enc_in.name: enc_input})
            dec_sess.run(None, {dec_in.name: dec_input})
            t1 = time.perf_counter_ns()
            if i >= self.warmup:
                stab_times.append(t1 - t0)

        ms_arr = np.array(stab_times, dtype=np.float64) / 1e6
        med = float(np.median(ms_arr))
        spikes = int(np.sum(ms_arr > 2 * med))
        jitter = float(np.std(np.diff(ms_arr)))
        max_val = float(np.max(ms_arr))

        rows = [
            ("Median latency (E2E)", _fmt_ms(med)),
            ("Latency spikes (>2x median)", f"{spikes} ({100 * spikes / len(ms_arr):.2f}%)"),
            ("Max latency", _fmt_ms(max_val)),
            ("Jitter (std of Δlatency)", _fmt_ms(jitter)),
            (f"Within {TARGET_DT_MS:.0f}ms budget", f"{'✅ YES' if max_val < TARGET_DT_MS else '❌ NO (max=' + _fmt_ms(max_val) + ')'}"),
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
        # 6. MEMORY
        # ══════════════════════════════════════════════════════════════
        gpu_peak, _ = _gpu_mem()
        results["memory"]["gpu_peak_mb"] = max(gpu_after, gpu_peak)

        print()
        print("┌" + "─" * W + "┐")
        print(f"│ 6. MEMORY".ljust(W + 1) + "│")
        print("├" + "─" * 42 + "┬" + "─" * 31 + "┤")
        mem_rows = [
            ("Model load time", f"{load_time:.2f}s"),
            ("RSS before → after", f"{rss_before:.1f} MB → {rss_after:.1f} MB (+{rss_after - rss_before:.1f})"),
            ("GPU before → after", f"{gpu_before:.0f} MB → {gpu_after:.0f} MB (+{gpu_after - gpu_before:.0f})"),
            ("GPU peak", f"{results['memory']['gpu_peak_mb']:.0f} MB / {gpu_total:.0f} MB"),
            ("Encoder model size", f"{self.encoder_path.stat().st_size / 1e6:.1f} MB"),
            ("Decoder model size", f"{self.decoder_path.stat().st_size / 1e6:.1f} MB"),
        ]
        for label, val in mem_rows:
            print(f"│ {label:<41s}│ {val:>30s} │")
        print("└" + "─" * 42 + "┴" + "─" * 31 + "┘")

        # ══════════════════════════════════════════════════════════════
        # Summary
        # ══════════════════════════════════════════════════════════════
        print()
        print("╔" + "═" * W + "╗")
        print("║" + "DEPLOYMENT READINESS SUMMARY".center(W) + "║")
        print("╠" + "═" * W + "╣")
        checks = [
            (f"E2E latency < 20ms (50Hz budget)", e2e_stats["p99_ms"] < TARGET_DT_MS,
             f"P99={_fmt_ms(e2e_stats['p99_ms'])}"),
            (f"E2E latency < 10ms (100Hz budget)", e2e_stats["p99_ms"] < 10.0,
             f"P99={_fmt_ms(e2e_stats['p99_ms'])}"),
            (f"E2E latency < 2ms (500Hz budget)", e2e_stats["p99_ms"] < 2.0,
             f"P99={_fmt_ms(e2e_stats['p99_ms'])}"),
            (f"No spikes > 2x median", spikes == 0,
             f"{spikes} spikes"),
            (f"Jitter < 1ms", jitter < 1.0,
             f"{_fmt_ms(jitter)}"),
            (f"GPU memory < 1GB", (gpu_after - gpu_before) < 1024,
             f"+{gpu_after - gpu_before:.0f}MB"),
        ]
        for label, ok, detail in checks:
            icon = "✅" if ok else "❌"
            print(f"║  {icon} {label:<40s} {detail:>{W - 46}s}  ║")
        print("╚" + "═" * W + "╝")
        results["deployment_checks"] = [
            {"check": c[0], "pass": c[1], "detail": c[2]} for c in checks
        ]

        # Save
        self._save(ts, results, ms_arr.tolist())
        return results

    def _save(self, ts: datetime, results: Dict, trace: List[float]) -> None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        stamp = ts.strftime("%Y%m%d_%H%M%S")

        # JSON
        jp = RESULTS_DIR / f"sonic_onnx_benchmark_{stamp}.json"
        with open(jp, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n📄 JSON saved: {jp}")

        # Markdown
        mp = RESULTS_DIR / f"sonic_onnx_benchmark_{stamp}.md"
        with open(mp, "w") as f:
            f.write(self._markdown(results))
        print(f"📝 Markdown saved: {mp}")

        # Plot
        if HAS_MPL and trace:
            pp = RESULTS_DIR / f"sonic_onnx_latency_{stamp}.png"
            self._plot(trace, pp, results)
            print(f"📊 Plot saved: {pp}")

    def _markdown(self, r: Dict) -> str:
        lines = [
            "# SONIC ONNX Inference Benchmark\n",
            f"**Date:** {r['timestamp']}  ",
            f"**GPU:** {r['system']['gpu']}  ",
            f"**Device:** {r['system']['device']}  ",
            f"**ORT:** {r['system']['onnxruntime']} | **NumPy:** {r['system']['numpy']}  ",
            f"**Model load:** {r.get('model_load_time_s', 0):.2f}s\n",
            "## Key Results\n",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Encoder latency (mean) | {_fmt_ms(r['encoder_latency']['mean_ms'])} |",
            f"| Decoder latency (mean) | {_fmt_ms(r['decoder_latency']['mean_ms'])} |",
            f"| E2E latency (mean) | {_fmt_ms(r['e2e_latency']['mean_ms'])} |",
            f"| E2E latency (P99) | {_fmt_ms(r['e2e_latency']['p99_ms'])} |",
            f"| Headroom vs 50Hz | {r.get('headroom_vs_50hz', 0):.0f}x |",
            f"| Sequential FPS | {_fmt_fps(r['throughput'][0]['fps'])} |",
        ]
        return "\n".join(lines) + "\n"

    def _plot(self, trace: List[float], path: Path, results: Dict) -> None:
        ms = np.array(trace)
        fig, axes = plt.subplots(2, 1, figsize=(14, 9))

        med = np.median(ms)
        p99 = np.percentile(ms, 99)

        ax = axes[0]
        ax.hist(ms, bins=min(200, max(50, len(ms) // 50)),
                color="#2196F3", alpha=0.85, edgecolor="none")
        ax.axvline(med, color="green", ls="--", lw=1.5,
                   label=f"Median: {_fmt_ms(med)}")
        ax.axvline(p99, color="red", ls="--", lw=1.5,
                   label=f"P99: {_fmt_ms(p99)}")
        ax.axvline(TARGET_DT_MS, color="orange", ls="-", lw=2,
                   label=f"Budget: {TARGET_DT_MS:.0f}ms")
        ax.set_xlabel("Latency (ms)")
        ax.set_ylabel("Count")
        ax.set_title(f"SONIC ONNX E2E Latency Distribution (N={len(ms):,}) — {results['system']['device']}")
        ax.legend()

        ax2 = axes[1]
        ax2.plot(ms, lw=0.3, color="#2196F3", alpha=0.7)
        ax2.axhline(med, color="green", ls="--", lw=1, alpha=0.8, label=f"Median: {_fmt_ms(med)}")
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
        description="SONIC ONNX Model Inference Benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--device", "-d", choices=["cuda", "cpu"], default="cuda",
                        help="Inference device")
    parser.add_argument("--iterations", "-n", type=int, default=10000,
                        help="Measurement iterations")
    parser.add_argument("--warmup", "-w", type=int, default=500,
                        help="Warmup iterations")
    parser.add_argument("--sustained", type=float, default=10.0,
                        help="Sustained throughput duration (seconds)")
    args = parser.parse_args()

    bench = SonicOnnxBenchmark(
        device=args.device,
        iterations=args.iterations,
        warmup=args.warmup,
        sustained_duration=args.sustained,
    )
    bench.run()


if __name__ == "__main__":
    main()
