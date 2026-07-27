#!/usr/bin/env python3
"""Low-latency 4-camera RTSP capture client for the RoboBaton_4p SC132 X5 demo.

One ffmpeg subprocess per camera pulls the RTSP stream with low-latency flags and
decodes it (VideoToolbox hwaccel on Apple Silicon) into raw BGR24 frames on stdout.
A reader thread per camera consumes exactly W*H*3 bytes per frame and records the
Mac-side arrival time of every frame.

What this measures (client/transport level):
  - per-camera decode fps vs the 60 fps target
  - frame-to-frame inter-arrival jitter
  - inter-camera arrival skew: how closely frames from different cameras arrive
    together on the Mac (pairwise nearest-neighbour |Δt| distribution)

What this does NOT measure:
  - sensor-level timestamps. camera_ts_ns (GPIO417 edge on CLOCK_MONOTONIC_RAW) and
    the IMU host_timestamp_ns live on the board clock and are emitted on the board's
    stdout, not in the RTSP stream. Those are handled by sync_verify.py against the
    board's `sensor_demo --diagnostics` + IMU output.

Usage:
  python3 capture_4cam.py --duration 10
  python3 capture_4cam.py --transport tcp --save-frames 5 --out runs/probe
"""
from __future__ import annotations

import argparse
import csv
import os
import signal
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

BOARD_IP_DEFAULT = "192.168.1.12"
PORTS_DEFAULT = [554, 555, 556, 557]
PATH_DEFAULT = "/PRR"
WIDTH, HEIGHT = 1280, 1088
FRAME_BYTES = WIDTH * HEIGHT * 3  # bgr24


@dataclass
class CamStats:
    cam_id: int
    frames: int = 0
    arrival_ns: list = field(default_factory=list)  # monotonic ns of each frame read completion
    bytes_in: int = 0


def build_ffmpeg_cmd(url: str, transport: str, hwaccel: bool) -> list[str]:
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
    ]
    if hwaccel:
        # VideoToolbox is faster but occasionally drops a frame on corrupt UDP packets;
        # software h264 decode on Apple Silicon handles 4x 1280x1088@60 with margin.
        cmd += ["-hwaccel", "videotoolbox"]
    # Low-latency ingest: nobuffer (demux) + low_delay (codec) + max_delay 0 shrink the
    # jitter buffer. -fps_mode passthrough emits each decoded frame immediately without
    # re-timing, which is what makes rawvideo pipe output actually flush.
    cmd += [
        "-rtsp_transport", transport,
        "-fflags", "nobuffer",
        "-flags", "low_delay",
        "-max_delay", "0",
        "-allowed_media_types", "video",
        "-i", url,
        "-an",
        "-fps_mode", "passthrough",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "pipe:1",
    ]
    return cmd


def reader_thread(cam_id: int, proc: subprocess.Popen, stats: CamStats,
                  stop_evt: threading.Event, save_frames: int,
                  out_dir: Path | None) -> None:
    """Read exactly FRAME_BYTES per frame until EOF or stop."""
    stream = proc.stdout
    assert stream is not None
    saved = 0
    try:
        while not stop_evt.is_set():
            blob = bytearray()
            remaining = FRAME_BYTES
            while remaining > 0:
                chunk = stream.read(remaining)
                if not chunk:
                    return  # EOF (ffmpeg exited)
                blob.extend(chunk)
                remaining -= len(chunk)
            stats.arrival_ns.append(time.monotonic_ns())
            stats.frames += 1
            stats.bytes_in += FRAME_BYTES
            if save_frames > 0 and saved < save_frames and out_dir is not None:
                (out_dir / f"cam{cam_id}_frame{saved:04d}.raw").write_bytes(bytes(blob))
                saved += 1
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[cam{cam_id}] reader error: {e}\n")


def nearest_neighbour_skew(per_cam_ns: dict[int, list[int]]) -> dict[tuple[int, int], float]:
    """For each ordered pair (a, b), mean of |arrival_a_k - nearest arrival_b| in ms."""
    result: dict[tuple[int, int], float] = {}
    ids = sorted(per_cam_ns)
    for a in ids:
        arr_a = per_cam_ns[a]
        for b in ids:
            if a == b:
                continue
            arr_b = per_cam_ns[b]
            diffs = []
            j = 0
            for ta in arr_a:
                # advance j to first b >= ta
                while j < len(arr_b) - 1 and arr_b[j] < ta:
                    j += 1
                best = abs(ta - arr_b[j])
                if j > 0:
                    best = min(best, abs(ta - arr_b[j - 1]))
                diffs.append(best)
            result[(a, b)] = (statistics.mean(diffs) / 1e6) if diffs else float("nan")
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=BOARD_IP_DEFAULT)
    ap.add_argument("--ports", type=int, nargs="+", default=PORTS_DEFAULT)
    ap.add_argument("--path", default=PATH_DEFAULT)
    ap.add_argument("--transport", choices=["udp", "tcp"], default="udp",
                    help="udp = lowest latency on a wired link (default); tcp = lossless")
    ap.add_argument("--hwaccel", action="store_true", help="enable VideoToolbox hwaccel (default off: software decode is robust)")
    ap.add_argument("--duration", type=float, default=10.0, help="capture seconds")
    ap.add_argument("--save-frames", type=int, default=0, help="save first N raw frames per cam")
    ap.add_argument("--out", default=None, help="output dir for frames + manifest csv")
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    urls = [f"rtsp://{args.ip}:{p}{args.path}" for p in args.ports]
    n_cams = len(urls)
    print(f"[capture] {n_cams} cams, transport={args.transport}, "
          f"hwaccel={args.hwaccel}, duration={args.duration}s, "
          f"res={WIDTH}x{HEIGHT}@60 target")

    procs: list[subprocess.Popen] = []
    stats: list[CamStats] = []
    threads: list[threading.Thread] = []
    stop_evt = threading.Event()

    # Start all ffmpeg processes
    for i, url in enumerate(urls):
        cmd = build_ffmpeg_cmd(url, args.transport, hwaccel=args.hwaccel)
        # Over-allocate pipe so a brief stall never deadlocks ffmpeg.
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             bufsize=FRAME_BYTES * 4)
        procs.append(p)
        s = CamStats(cam_id=i)
        stats.append(s)
        t = threading.Thread(target=reader_thread, args=(i, p, s, stop_evt,
                                                         args.save_frames, out_dir),
                             daemon=True)
        threads.append(t)
        t.start()
        print(f"  cam{i} -> {url}  (pid {p.pid})")

    def _bail(*_):
        stop_evt.set()
    signal.signal(signal.SIGINT, _bail)

    t0 = time.monotonic()
    while time.monotonic() - t0 < args.duration and not stop_evt.is_set():
        time.sleep(0.2)
    stop_evt.set()

    # Stop ffmpeg (SIGINT lets it close RTSP cleanly); then join readers.
    for p in procs:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
    deadline = time.monotonic() + 3.0
    for p in procs:
        rem = max(0.1, deadline - time.monotonic())
        try:
            p.wait(timeout=rem)
        except subprocess.TimeoutExpired:
            p.kill()
    for t in threads:
        t.join(timeout=2.0)

    # ---- Report ----
    print("\n=== per-camera ===")
    for s in stats:
        arr = s.arrival_ns
        fps = (len(arr) - 1) / ((arr[-1] - arr[0]) / 1e9) if len(arr) > 1 else 0.0
        if len(arr) > 1:
            dts = [(arr[k + 1] - arr[k]) / 1e6 for k in range(len(arr) - 1)]
            jitter_ms = statistics.pstdev(dts)
            dt_mean_ms = statistics.mean(dts)
        else:
            jitter_ms = dt_mean_ms = float("nan")
        print(f"  cam{s.cam_id}: frames={s.frames:5d}  fps={fps:5.2f}  "
              f"dt_mean={dt_mean_ms:6.2f}ms  jitter(pstdev)={jitter_ms:5.2f}ms  "
              f"MB_in={s.bytes_in/1e6:7.1f}")

    # Inter-camera nearest-neighbour arrival skew
    per_cam = {s.cam_id: s.arrival_ns for s in stats if s.arrival_ns}
    if len(per_cam) >= 2:
        print("\n=== inter-camera arrival skew (mean |Δt| to nearest frame, ms) ===")
        skew = nearest_neighbour_skew(per_cam)
        all_pair_means = []
        ids = sorted(per_cam)
        header = "      " + " ".join(f"cam{b:<4d}" for b in ids)
        print(header)
        for a in ids:
            row = [f"cam{a:<2d}->"]
            for b in ids:
                if a == b:
                    row.append("  —   ")
                else:
                    v = skew[(a, b)]
                    all_pair_means.append(v)
                    row.append(f"{v:5.2f} ")
            print(" ".join(row))
        if all_pair_means:
            print(f"  overall mean pairwise arrival skew = {statistics.mean(all_pair_means):.2f} ms, "
                  f"max = {max(all_pair_means):.2f} ms")

    if out_dir:
        manifest = out_dir / "arrival_manifest.csv"
        with manifest.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cam_id", "frame_idx", "arrival_monotonic_ns"])
            for s in stats:
                for idx, ns in enumerate(s.arrival_ns):
                    w.writerow([s.cam_id, idx, ns])
        print(f"\n[manifest] {manifest}")
        if args.save_frames:
            print(f"[frames]   {out_dir}/camN_frameXXXX.raw  ({WIDTH}x{HEIGHT} bgr24, "
                  f"view with: ffmpeg -f rawvideo -pix_fmt bgr24 -s {WIDTH}x{HEIGHT} -i cam0_frame0000.raw out.png)")

    # Non-zero exit if any cam clearly failed
    failed = [s.cam_id for s in stats if s.frames == 0]
    if failed:
        print(f"\n[error] cams with no frames: {failed}", file=sys.stderr)
        for p, i in zip(procs, range(n_cams)):
            err = p.stderr.read().decode("utf-8", "replace") if p.stderr else ""
            if err:
                print(f"--- ffmpeg cam{i} stderr ---\n{err[:1500]}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
