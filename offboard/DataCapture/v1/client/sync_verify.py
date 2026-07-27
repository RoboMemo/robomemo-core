#!/usr/bin/env python3
"""Sensor-level time-sync verification for the RoboBaton_4p SC132 + ICM42688 X5 demo.

Parses the board's stdout from two concurrent board-side runs (both timestamped on the
board's CLOCK_MONOTONIC_RAW — the shared timebase) and reports three things:

  (1) 4-camera INTRA-group skew  — from `cam_demo --diagnostics` `frameset ... calc_skew_ns`
      lines. This is the hardware sync guarantee; design max is 2 ms (SC132 header),
      measured ~1.06 ms.
  (2) Per-camera send health     — fps, queue_full_rejects (must be 0), pipeline_delay_ms,
      send_max_ms from the `cam%d ...` lines.
  (3) cam↔IMU shared-clock phase  — for each camera group_ts_ns, |Δ| to the nearest IMU
      ts_ns. Because both clocks are CLOCK_MONOTONIC_RAW, this is bounded by half the IMU
      sample period (~0.5 ms at 1 kHz). It proves a common timebase, NOT the physical
      sensor time-delay. Use motion_xcorr.py for the physical TD (needs a motion event).

Inputs (one or two files; if a single file contains both line types, it is split automatically):
  --cam-log    path to `cam_demo --diagnostics` stdout
  --imu-log    path to `imu_reader_demo --sample-rate-hz 1000 --print-rate-hz 1000` stdout

Usage:
  python3 sync_verify.py --cam-log board_cam.log --imu-log board_imu.log
  python3 sync_verify.py --combined board_sensor.log
"""
from __future__ import annotations

import argparse
import re
import statistics
import sys
from dataclasses import dataclass, field


# ---- Parsers ----------------------------------------------------------------

FRAMESET_RE = re.compile(
    r"frameset group_id=(\d+) group_ts_ns=(\d+) group_skew_ns=(\d+) calc_skew_ns=(\d+)"
    r"(?P<cams>(?: cam\d+\([^)]*\))*)"
)
CAMITEM_RE = re.compile(r"cam(\d+)\(seq=(\d+),frame_id=(\d+),camera_ts_ns=(\d+)\)")
CAMLINE_RE = re.compile(
    r"cam(\d+) fps=([\d.]+) last_seq=(\d+) group_id=(\d+) group_skew_ns=(\d+) "
    r"queue=(\d+)/(\d+) (?:queue_full_rejects|full_waits)=(\d+) pipeline_delay_ms=(\d+) "
    r"camera_ts_ns=(\d+) rtsp_ts_ns=(\d+) send_avg_ms=([\d.]+) send_max_ms=([\d.]+)"
)
IMU_RE = re.compile(
    r"ts_ns=(\d+)\s+dt_ms=([\-\d.eE]+)\s+temp_c=([\-\d.eE]+)\s+"
    r"accel_mps2=\[([\-\d.eE]+),\s*([\-\d.eE]+),\s*([\-\d.eE]+)\]\s+"
    r"accel_norm_mps2=([\-\d.eE]+)\s+gyro_rps=\[([\-\d.eE]+),\s*([\-\d.eE]+),\s*([\-\d.eE]+)\]"
)
SENSOR_RESULT_RE = re.compile(
    r"SENSOR_IMU_RESULT samples=(\d+) invalid=(\d+) timestamp_duplicates=(\d+) "
    r"timestamp_regressions=(\d+) effective_hz=([\d.]+) min_dt_ns=(\d+) max_dt_ns=(\d+)"
)


@dataclass
class FrameSet:
    group_id: int
    group_ts_ns: int
    group_skew_ns: int            # configured threshold (default 2 000 000)
    calc_skew_ns: int             # actual max-min of the 4 camera_ts_ns
    cam_ts_ns: dict[int, int] = field(default_factory=dict)  # per-cam timestamp


@dataclass
class CamSendStat:
    cam_id: int
    fps: float
    group_skew_ns: int
    queue_full_rejects: int
    pipeline_delay_ms: int
    camera_ts_ns: int
    send_avg_ms: float
    send_max_ms: float


def parse_lines(text: str):
    """Yield parsed records from a combined or single stream."""
    framesets: list[FrameSet] = []
    sendstats: list[CamSendStat] = []
    imu: list[dict] = []
    sensor_result = None
    for line in text.splitlines():
        if line.startswith("frameset "):
            m = FRAMESET_RE.search(line)
            if not m:
                continue
            fs = FrameSet(
                group_id=int(m.group(1)),
                group_ts_ns=int(m.group(2)),
                group_skew_ns=int(m.group(3)),
                calc_skew_ns=int(m.group(4)),
            )
            for cm in CAMITEM_RE.finditer(m.group("cams")):
                fs.cam_ts_ns[int(cm.group(1))] = int(cm.group(4))
            framesets.append(fs)
        elif line.startswith("cam") and "fps=" in line:
            m = CAMLINE_RE.search(line)
            if m:
                sendstats.append(CamSendStat(
                    cam_id=int(m.group(1)), fps=float(m.group(2)),
                    group_skew_ns=int(m.group(5)), queue_full_rejects=int(m.group(8)),
                    pipeline_delay_ms=int(m.group(9)), camera_ts_ns=int(m.group(10)),
                    send_avg_ms=float(m.group(12)), send_max_ms=float(m.group(13)),
                ))
        elif line.startswith("ts_ns="):
            m = IMU_RE.search(line)
            if m:
                imu.append({
                    "ts_ns": int(m.group(1)),
                    "dt_ms": float(m.group(2)),
                    "accel": [float(m.group(4)), float(m.group(5)), float(m.group(6))],
                    "accel_norm": float(m.group(7)),
                    "gyro": [float(m.group(8)), float(m.group(9)), float(m.group(10))],
                })
        elif line.startswith("SENSOR_IMU_RESULT"):
            sensor_result = SENSOR_RESULT_RE.search(line)
    return framesets, sendstats, imu, sensor_result


def _summarize_ms(label: str, vals_ms):
    if not vals_ms:
        print(f"  {label}: (no data)")
        return
    vals_ms = sorted(vals_ms)
    def pct(p):
        return vals_ms[min(len(vals_ms) - 1, int(round(p * (len(vals_ms) - 1))))]
    print(f"  {label}: n={len(vals_ms)} mean={statistics.mean(vals_ms):.3f}ms "
          f"p50={pct(0.50):.3f} p95={pct(0.95):.3f} p99={pct(0.99):.3f} "
          f"max={vals_ms[-1]:.3f}ms")


def nearest_diff_ns(sorted_ts: list[int], t: int) -> int:
    """Binary search nearest value in sorted_ts; return abs diff in ns."""
    import bisect
    i = bisect.bisect_left(sorted_ts, t)
    best = abs(t - sorted_ts[i]) if i < len(sorted_ts) else None
    if i > 0:
        cand = abs(t - sorted_ts[i - 1])
        best = cand if best is None else min(best, cand)
    return best  # type: ignore[return-value]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam-log")
    ap.add_argument("--imu-log")
    ap.add_argument("--combined", help="single file containing both cam diagnostics and IMU lines")
    args = ap.parse_args()

    framesets, sendstats, imu, sensor_result = [], [], [], None
    if args.combined:
        fs, ss, im, sr = parse_lines(Path_read(args.combined))
        framesets, sendstats, imu, sensor_result = fs, ss, im, sr
    else:
        if args.cam_log:
            fs, ss, _, _ = parse_lines(Path_read(args.cam_log))
            framesets, sendstats = fs, ss
        if args.imu_log:
            _, _, im, sr = parse_lines(Path_read(args.imu_log))
            imu, sensor_result = im, sr

    if not (framesets or imu):
        print("No frameset or IMU records parsed. Check log format / file paths.", file=sys.stderr)
        return 2

    # ---- (1) 4-cam intra-group skew ----
    print("=" * 68)
    print("(1) 4-CAMERA INTRA-GROUP SYNC  (calc_skew_ns = max−min of the 4 camera_ts_ns)")
    print("=" * 68)
    if framesets:
        skew_ms = [fs.calc_skew_ns / 1e6 for fs in framesets]
        _summarize_ms("calc_skew", skew_ms)
        over = sum(1 for s in skew_ms if s > 2.0)
        print(f"  groups exceeding 2.0 ms design limit: {over} / {len(skew_ms)}")
        if framesets[0].cam_ts_ns:
            print(f"  example group_id={framesets[-1].group_id}: "
                  + " ".join(f"cam{k}={v}" for k, v in sorted(framesets[-1].cam_ts_ns.items())))
    else:
        print("  (no `frameset` lines found in cam log)")

    # ---- (2) per-camera send health ----
    print("\n" + "=" * 68)
    print("(2) PER-CAMERA RTSP SEND HEALTH")
    print("=" * 68)
    by_cam: dict[int, list[CamSendStat]] = {}
    for s in sendstats:
        by_cam.setdefault(s.cam_id, []).append(s)
    for cam_id in sorted(by_cam):
        rows = by_cam[cam_id]
        fps_mean = statistics.mean(r.fps for r in rows)
        rejects = sum(r.queue_full_rejects for r in rows)
        pd_max = max(r.pipeline_delay_ms for r in rows)
        sm_max = max(r.send_max_ms for r in rows)
        print(f"  cam{cam_id}: fps_mean={fps_mean:6.2f}  queue_full_rejects(sum)={rejects}  "
              f"pipeline_delay_ms(max)={pd_max}  send_max_ms(max)={sm_max:.2f}")

    # ---- (3) cam↔IMU shared-clock phase ----
    print("\n" + "=" * 68)
    print("(3) CAM↔IMU SHARED-CLOCK PHASE  (|group_ts_ns − nearest IMU ts_ns|)")
    print("    Proves a common CLOCK_MONOTONIC_RAW timebase; NOT the physical sensor TD.")
    print("=" * 68)
    if framesets and imu:
        imu_ts = sorted(s["ts_ns"] for s in imu)
        phase_ms = [nearest_diff_ns(imu_ts, fs.group_ts_ns) / 1e6 for fs in framesets]
        _summarize_ms("phase(group_ts vs nearest IMU)", phase_ms)
        print(f"  IMU samples parsed: {len(imu)}; cam groups parsed: {len(framesets)}")
        # IMU rate sanity
        if len(imu) > 2:
            span_s = (imu_ts[-1] - imu_ts[0]) / 1e9
            print(f"  IMU span={span_s:.2f}s  effective rate≈{(len(imu_ts)-1)/span_s:.1f} Hz "
                  f"(print-rate limited; acquisition still 1 kHz)")
    else:
        missing = []
        if not framesets:
            missing.append("camera frameset log")
        if not imu:
            missing.append("IMU sample log")
        print(f"  (skipped — need {' + '.join(missing)})")

    if sensor_result:
        print("\n" + "=" * 68)
        print("SENSOR_IMU_RESULT (board-reported IMU acquisition summary)")
        print("=" * 68)
        print(f"  samples={sensor_result.group(1)} invalid={sensor_result.group(2)} "
              f"duplicates={sensor_result.group(3)} regressions={sensor_result.group(4)} "
              f"effective_hz={sensor_result.group(5)}")
        print(f"  min_dt_ns={sensor_result.group(6)} max_dt_ns={sensor_result.group(7)}")

    print("\nNotes:")
    print("  - 4-cam calc_skew is the hardware sync guarantee; expect p99 < 2 ms.")
    print("  - cam↔IMU phase < ~0.5 ms confirms shared CLOCK_MONOTONIC_RAW; physical TD")
    print("    (exposure + ISP + IMU group delay) must be measured via a motion event")
    print("    (see motion_xcorr.py).")
    return 0


def Path_read(p: str) -> str:
    from pathlib import Path
    return Path(p).read_text(errors="replace")


if __name__ == "__main__":
    sys.exit(main())
