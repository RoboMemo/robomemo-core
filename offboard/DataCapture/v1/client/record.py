#!/usr/bin/env python3
"""X5 4-camera + IMU data recorder (Phase 1).

Records one episode to a self-contained directory:
  cam0.mp4 .. cam3.mp4   - RTSP H.264 remuxed losslessly (ffmpeg -c copy, TCP)
  imu.jsonl              - per-sample {ts_ns, accel_mps2[3], gyro_rps[3], temp_c} @ 1 kHz
  cam_diag.log, imu.log  - raw board stdout (board CLOCK_MONOTONIC_RAW timestamps)
  manifest.json         - dual-clock recording window + per-cam stats + board anchors

The board side runs ONE cam_demo --diagnostics (serves RTSP + emits frameset group_ts on
the same process, so recorded frames align by index to board timestamps) plus imu_reader_demo.

Dual timestamps:
  - Board sensor clock: group_ts_ns (CLOCK_MONOTONIC_RAW) from cam_diag; per-frame
    reconstructable as anchor_ns + (k - anchor_k)/fps (software_gpio trigger is periodic).
  - Mac arrival clock:   time.time_ns() window + per-packet pts_time (ffprobe post-hoc).

Usage:
  python3 record.py --dataset grasp_001 --duration 60 --out episodes
  python3 record.py --dataset grasp_001            # Ctrl-C to stop
"""
from __future__ import annotations

import argparse
import json
import queue
import re
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

BOARD_IP_DEFAULT = "192.168.1.12"
PORTS_DEFAULT = [554, 555, 556, 557]
PATH_DEFAULT = "/PRR"
WIDTH, HEIGHT, FPS = 1280, 1088, 60
SSH_OPTS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=5"]

FRAMESET_RE = re.compile(
    r"frameset group_id=(\d+) group_ts_ns=(\d+) group_skew_ns=(\d+) calc_skew_ns=(\d+)")
IMU_RE = re.compile(
    r"ts_ns=(\d+)\s+dt_ms=([\-\d.eE]+)\s+temp_c=([\-\d.eE]+)\s+"
    r"accel_mps2=\[([\-\d.eE]+),\s*([\-\d.eE]+),\s*([\-\d.eE]+)\]\s+"
    r"accel_norm_mps2=([\-\d.eE]+)\s+gyro_rps=\[([\-\d.eE]+),\s*([\-\d.eE]+),\s*([\-\d.eE]+)\]")


def ssh(ip: str, cmd: str, timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", *SSH_OPTS, f"root@{ip}", cmd],
                          capture_output=True, text=True, timeout=timeout)


def ensure_board(ip: str) -> None:
    """Ensure cam_demo --diagnostics + imu_reader are running on the board. Idempotent:
    if both are already up, skip the restart (fast path). Otherwise restart (~10s)."""
    check = ssh(ip,
                "pgrep -f 'cam_demo --diagnostics' >/dev/null && "
                "pgrep -f bin/imu_reader_demo >/dev/null && echo ALREADY",
                timeout=8)
    if "ALREADY" in (check.stdout or ""):
        print(f"[board] cam_demo --diagnostics + imu_reader already running on {ip} (fast path)")
        return
    print(f"[board] starting cam_demo --diagnostics + imu_reader on {ip}")
    ssh(ip, """
set +e
cd /root/demo
killall -q cam_demo imu_reader_demo sensor_demo 2>/dev/null
sleep 1
/etc/init.d/S90cam-service start 2>/dev/null || true
sleep 1
: > /root/demo/cam_diag.log
: > /root/demo/imu.log
nohup ./cam_demo --diagnostics > /root/demo/cam_diag.log 2>&1 &
disown
sleep 3
nohup ./imu_reader_demo --sample-rate-hz 1000 --count 0 > /root/demo/imu.log 2>&1 &
disown
sleep 2
echo "cam=$(pgrep -f bin/cam_demo | head -1) imu=$(pgrep -f bin/imu_reader_demo | head -1)"
""", timeout=30)
    # wait for RTSP ports
    import socket
    for p in PORTS_DEFAULT:
        for _ in range(20):
            try:
                with socket.create_connection((ip, p), timeout=1):
                    break
            except OSError:
                time.sleep(0.5)
        else:
            print(f"[warn] port {p} not up on board; recording may fail", file=sys.stderr)


def truncate_board_logs(ip: str) -> None:
    """Clear board logs right before recording so board-clock anchors span only the
    recording window (aligns board window with the Mac recording window)."""
    ssh(ip, ": > /root/demo/cam_diag.log; : > /root/demo/imu.log", timeout=10)


def restore_board(ip: str) -> None:
    """Leave the board with a plain cam_demo serving RTSP."""
    ssh(ip, """
set +e
cd /root/demo
killall -q cam_demo imu_reader_demo 2>/dev/null
sleep 1
nohup ./cam_demo > /root/demo/cam_demo.log 2>&1 &
disown
""", timeout=20)


def ffmpeg_record_cmd(url: str, out_mp4: Path) -> list[str]:
    # TCP for lossless recording (no UDP packet loss); -c copy remuxes the board's H.264
    # straight into mp4 with no decode and no quality loss.
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "warning",
        "-rtsp_transport", "tcp",
        "-i", url,
        "-c", "copy",
        "-y",
        "-f", "mp4", str(out_mp4),
    ]


def finalize(out_dir: Path, episode_id: str, ip: str, ports: list[int],
             mac_start_ns: int, mac_end_ns: int, started_iso: str) -> dict:
    """scp board logs, parse IMU -> jsonl, ffprobe mp4s, build manifest."""
    print("[finalize] fetching board logs")
    cam_log = out_dir / "cam_diag.log"
    imu_log = out_dir / "imu.log"
    subprocess.run(["scp", *SSH_OPTS, f"root@{ip}:/root/demo/cam_diag.log", str(cam_log)],
                   capture_output=True, timeout=30)
    subprocess.run(["scp", *SSH_OPTS, f"root@{ip}:/root/demo/imu.log", str(imu_log)],
                   capture_output=True, timeout=60)

    # Parse frameset anchors
    anchors, skew_vals = [], []
    for line in cam_log.read_text(errors="replace").splitlines():
        m = FRAMESET_RE.search(line)
        if m:
            anchors.append({
                "group_id": int(m.group(1)),
                "group_ts_ns": int(m.group(2)),
                "calc_skew_ns": int(m.group(4)),
            })
            skew_vals.append(int(m.group(4)))

    # Parse IMU -> jsonl
    imu_jsonl = out_dir / "imu.jsonl"
    n_imu = 0
    imu_first = imu_last = None
    with imu_jsonl.open("w") as fout:
        for line in imu_log.read_text(errors="replace").splitlines():
            m = IMU_RE.search(line)
            if not m:
                continue
            rec = {
                "ts_ns": int(m.group(1)),
                "temp_c": float(m.group(3)),
                "accel_mps2": [float(m.group(4)), float(m.group(5)), float(m.group(6))],
                "gyro_rps": [float(m.group(8)), float(m.group(9)), float(m.group(10))],
            }
            fout.write(json.dumps(rec) + "\n")
            if imu_first is None:
                imu_first = rec["ts_ns"]
            imu_last = rec["ts_ns"]
            n_imu += 1

    # ffprobe each mp4
    cams = []
    for i, _ in enumerate(ports):
        mp4 = out_dir / f"cam{i}.mp4"
        if not mp4.exists():
            cams.append({"id": i, "mp4": mp4.name, "error": "missing"})
            continue
        info = {"id": i, "mp4": mp4.name, "size_bytes": mp4.stat().st_size}
        pp = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", "-rtsp_transport", "tcp",
             "-select_streams", "v:0",
             "-show_entries", "stream=codec_name,width,height,r_frame_rate:format=duration",
             "-of", "json", str(mp4)],
            capture_output=True, text=True)
        try:
            j = json.loads(pp.stdout)
            s = j.get("streams", [{}])[0]
            info.update({"codec": s.get("codec_name"),
                         "width": s.get("width"), "height": s.get("height"),
                         "fps": s.get("r_frame_rate"),
                         "duration_s": float(j.get("format", {}).get("duration", 0))})
        except Exception:
            info["ffprobe_error"] = pp.stderr[:200]
        # frame count via packet count
        pn = subprocess.run(
            ["ffprobe", "-hide_banner", "-v", "error", "-select_streams", "v:0",
             "-count_packets", "-show_entries", "stream=nb_read_packets",
             "-of", "csv=p=0", str(mp4)],
            capture_output=True, text=True)
        try:
            info["frame_count"] = int(pn.stdout.strip())
        except ValueError:
            info["frame_count"] = None
        cams.append(info)

    board_start = anchors[0]["group_ts_ns"] if anchors else None
    board_end = anchors[-1]["group_ts_ns"] if anchors else None
    manifest = {
        "episode_id": episode_id,
        "dataset": None,
        "started_at_mac_iso": started_iso,
        "recording_window": {
            "mac": {"start_ns": mac_start_ns, "end_ns": mac_end_ns,
                    "duration_s": (mac_end_ns - mac_start_ns) / 1e9},
            "board": {"start_ns": board_start, "end_ns": board_end,
                      "duration_s": ((board_end - board_start) / 1e9) if board_start and board_end else None,
                      "clock": "CLOCK_MONOTONIC_RAW"},
        },
        "cams": cams,
        "imu": {"sample_count": n_imu, "first_ts_ns": imu_first, "last_ts_ns": imu_last,
                "rate_hz": ((n_imu - 1) / ((imu_last - imu_first) / 1e9)
                            if imu_first and imu_last and imu_last > imu_first else None),
                "file": "imu.jsonl"},
        "board_anchors": anchors,
        "calc_skew_ns": {
            "n": len(skew_vals),
            "mean": round(statistics.mean(skew_vals)) if skew_vals else None,
            "max": max(skew_vals) if skew_vals else None,
        },
        "notes": ("Per-frame board timestamp = anchor.group_ts_ns + (frame_k - anchor_k)/fps "
                  "(software_gpio trigger is periodic at the camera fps). Mac per-frame arrival "
                  "= mac.start_ns + packet_pts_time_s*1e9 (ffprobe -show_packets)."),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


# ----- --stream: live preview to the Platform collection page -----

class Streamer:
    """Asyncio websockets client on a background thread. Other threads push messages
    via send() (thread-safe, non-blocking, drops on backpressure — preview is lossy)."""

    def __init__(self, ws_url: str):
        self.ws_url = ws_url
        self.q: "queue.Queue[object]" = queue.Queue(maxsize=256)
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.ready = threading.Event()
        self.alive = False

    def start(self) -> None:
        self.thread.start()
        self.ready.wait(timeout=5.0)
        if self.alive:
            print(f"[stream] WS connected to {self.ws_url}")
        else:
            print("[stream] WS did not connect (backend down?) — recording continues", file=sys.stderr)

    def _run(self) -> None:
        import asyncio
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._main())
        except Exception as e:
            sys.stderr.write(f"[stream] WS error: {e}\n")
        finally:
            self.ready.set()

    async def _main(self) -> None:
        import websockets
        async with websockets.connect(self.ws_url, max_size=16 * 1024 * 1024) as ws:
            await ws.send(json.dumps({"type": "register", "clientType": "recorder"}))
            self.alive = True
            self.ready.set()
            while True:
                try:
                    msg = self.q.get(timeout=0.5)
                except queue.Empty:
                    continue
                if msg is None:
                    return
                await ws.send(json.dumps(msg))

    def send(self, msg: dict) -> None:
        if not self.alive:
            return
        try:
            self.q.put_nowait(msg)
        except queue.Full:
            pass

    def stop(self) -> None:
        if self.alive:
            self.q.put(None)
        self.thread.join(timeout=3.0)
        self.alive = False


def ffmpeg_record_and_preview_cmd(url: str, mp4_path: Path, fps: int) -> list[str]:
    """One RTSP connection, two outputs: lossless mp4 remux + downscaled mjpeg preview.
    Using a single client per RTSP port keeps the board's RTSP server happy (it serves
    one client per port well; a second client throttles badly)."""
    return [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-hwaccel", "videotoolbox",
        "-rtsp_transport", "tcp", "-fflags", "nobuffer", "-flags", "low_delay",
        "-i", url,
        "-c", "copy", "-f", "mp4", "-y", str(mp4_path),
        "-vf", f"fps={fps},scale=480:-1", "-q:v", "4", "-f", "mjpeg", "pipe:1",
    ]


def preview_reader_thread(cam_id: int, stdout, streamer: "Streamer",
                          stop_evt: dict) -> None:
    """Read mjpeg frames from a dual-output ffmpeg's stdout, push as sensor_data camN."""
    import base64
    buf = bytearray()
    soi_marker, eoi_marker = b"\xff\xd8", b"\xff\xd9"
    try:
        while not stop_evt["flag"]:
            chunk = stdout.read(65536)
            if not chunk:
                break
            buf.extend(chunk)
            while True:
                soi = buf.find(soi_marker)
                if soi < 0:
                    buf.clear()
                    break
                eoi = buf.find(eoi_marker, soi + 2)
                if eoi < 0:
                    if soi > 0:
                        del buf[:soi]
                    break
                frame = bytes(buf[soi:eoi + 2])
                del buf[:eoi + 2]
                b64 = base64.b64encode(frame).decode("ascii")
                streamer.send({
                    "type": "sensor_data", "sensorType": f"cam{cam_id}",
                    "image": f"data:image/jpeg;base64,{b64}",
                })
    except Exception as e:
        sys.stderr.write(f"[stream] cam{cam_id} preview error: {e}\n")


def imu_tail_thread(ip: str, streamer: "Streamer", stop_evt: dict) -> None:
    """Tail board imu.log live; push ~50 Hz of accel/gyro to the browser (1000 Hz local)."""
    proc = subprocess.Popen(
        ["ssh", *SSH_OPTS, f"root@{ip}", "tail -n +1 -f /root/demo/imu.log"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    n = 0
    try:
        for line in proc.stdout:
            if stop_evt["flag"]:
                break
            m = IMU_RE.search(line)
            if not m:
                continue
            n += 1
            if n % 20:  # 1000 Hz -> 50 Hz
                continue
            streamer.send({
                "type": "sensor_data", "sensorType": "imu",
                "ts_ns": int(m.group(1)), "temp_c": float(m.group(3)),
                "accel_mps2": [float(m.group(4)), float(m.group(5)), float(m.group(6))],
                "gyro_rps": [float(m.group(8)), float(m.group(9)), float(m.group(10))],
            })
    except Exception as e:
        sys.stderr.write(f"[stream] imu tail error: {e}\n")
    finally:
        proc.terminate()


def post_episode(api_base: str, dataset: str, episode_id: str, manifest: dict,
                 out_dir: Path) -> None:
    """Best-effort: register the episode with the Platform backend."""
    import urllib.request
    cams = manifest.get("cams", [])
    payload = {
        "id": episode_id,
        "datasetId": dataset or "rdk_x5",
        "name": episode_id,
        "frameCount": cams[0].get("frame_count") if cams else None,
        "duration": manifest["recording_window"]["mac"]["duration_s"],
        "fps": FPS,
        "source": "rdk_x5",
        "episodeDir": str(out_dir),
        "camPaths": [c.get("mp4") for c in cams],
        "imuPath": "imu.jsonl",
        "manifest": "manifest.json",
        "calc_skew_ns": manifest.get("calc_skew_ns"),
    }
    req = urllib.request.Request(
        f"{api_base}/api/episodes", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            print(f"[stream] episode registered: {json.loads(r.read())['id']}")
    except Exception as e:
        sys.stderr.write(f"[stream] episode register failed (non-fatal): {e}\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ip", default=BOARD_IP_DEFAULT)
    ap.add_argument("--ports", type=int, nargs="+", default=PORTS_DEFAULT)
    ap.add_argument("--path", default=PATH_DEFAULT)
    ap.add_argument("--dataset", default="episode", help="dataset/scene name (used in episode dir)")
    ap.add_argument("--duration", type=float, default=0.0, help="seconds; 0 = run until Ctrl-C")
    ap.add_argument("--out", default="episodes", help="output root dir")
    ap.add_argument("--stream", action="store_true",
                    help="live-preview 4 cams + IMU to the Platform collection page during recording")
    ap.add_argument("--ws-url", default="ws://localhost:3001")
    ap.add_argument("--api-base", default="http://localhost:3001")
    ap.add_argument("--preview-fps", type=int, default=10)
    ap.add_argument("--episode-id", default=None,
                    help="use this episode id/dir name instead of <dataset>_<timestamp>")
    ap.add_argument("--keep-board", action="store_true",
                    help="leave cam_demo --diagnostics + imu_reader running after stop (UI flow: fast re-record)")
    args = ap.parse_args()

    import datetime
    episode_id = args.episode_id or f"{args.dataset}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    out_dir = Path(args.out) / episode_id
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[record] episode={episode_id}  dir={out_dir}  duration="
          f"{'∞' if args.duration == 0 else args.duration}s")

    # Install interrupt handlers before board setup so Ctrl-C during the ~10s board
    # restart is graceful instead of an ugly KeyboardInterrupt traceback.
    stop_evt = {"flag": False}

    def _stop(*_):
        stop_evt["flag"] = True
    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    try:
        ensure_board(args.ip)
        truncate_board_logs(args.ip)
    except KeyboardInterrupt:
        print("[record] interrupted during board setup", file=sys.stderr)
        return 1

    urls = [f"rtsp://{args.ip}:{p}{args.path}" for p in args.ports]
    procs = []
    for i, url in enumerate(urls):
        mp4 = out_dir / f"cam{i}.mp4"
        if args.stream:
            # one connection per cam → mp4 (lossless remux) + mjpeg preview (stdout)
            p = subprocess.Popen(ffmpeg_record_and_preview_cmd(url, mp4, args.preview_fps),
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            p = subprocess.Popen(ffmpeg_record_cmd(url, mp4),
                                 stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        procs.append(p)
        print(f"  cam{i} -> {mp4.name}  (pid {p.pid})")

    streamer = None
    preview_threads: list[threading.Thread] = []
    imu_thread = None
    if args.stream:
        streamer = Streamer(args.ws_url)
        streamer.start()
        streamer.send({"type": "recording_status", "state": "recording",
                       "episodeId": episode_id, "dataset": args.dataset})
        for i in range(len(urls)):
            t = threading.Thread(target=preview_reader_thread,
                                 args=(i, procs[i].stdout, streamer, stop_evt), daemon=True)
            t.start()
            preview_threads.append(t)
        imu_thread = threading.Thread(target=imu_tail_thread,
                                      args=(args.ip, streamer, stop_evt), daemon=True)
        imu_thread.start()
        print(f"[stream] preview 4 cams @ {args.preview_fps}fps + IMU @50Hz")

    mac_start_ns = time.time_ns()
    started_iso = datetime.datetime.now().isoformat()
    print(f"[record] recording... (Ctrl-C to stop)" if args.duration == 0
          else f"[record] recording for {args.duration}s")

    t0 = time.monotonic()
    try:
        while not stop_evt["flag"]:
            if args.duration > 0 and (time.monotonic() - t0) >= args.duration:
                break
            # detect early ffmpeg death
            for p in procs:
                if p.poll() is not None:
                    raise RuntimeError(f"ffmpeg exited early (code {p.returncode})")
            time.sleep(0.2)
    except RuntimeError as e:
        print(f"[error] {e}", file=sys.stderr)
    mac_end_ns = time.time_ns()

    # Stop ffmpeg with SIGINT so it finalizes the mp4 moov atom
    print("[record] stopping ffmpeg (finalizing mp4)")
    for p in procs:
        if p.poll() is None:
            try:
                p.send_signal(signal.SIGINT)
            except Exception:
                pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()
    for p in procs:
        if p.stderr:
            err = p.stderr.read()
            if isinstance(err, bytes):
                err = err.decode("utf-8", "replace")
            if err.strip():
                print(f"  ffmpeg stderr: {err[:300]}", file=sys.stderr)

    manifest = finalize(out_dir, episode_id, args.ip, args.ports, mac_start_ns, mac_end_ns, started_iso)
    if not args.keep_board:
        restore_board(args.ip)

    if streamer is not None:
        for t in preview_threads:
            t.join(timeout=2)
        if imu_thread:
            imu_thread.join(timeout=2)
        streamer.send({"type": "recording_status", "state": "stopped", "episodeId": episode_id})
        post_episode(args.api_base, args.dataset, episode_id, manifest, out_dir)
        streamer.stop()

    # Summary
    print("\n=== episode summary ===")
    print(f"  dir: {out_dir}")
    for c in manifest["cams"]:
        print(f"  cam{c['id']}: {c.get('frame_count')} frames, {c.get('codec')}, "
              f"{c.get('duration_s','?')}s, {c.get('size_bytes',0)/1e6:.1f}MB")
    im = manifest["imu"]
    print(f"  imu: {im['sample_count']} samples, ~{im['rate_hz']:.1f} Hz" if im["rate_hz"] else f"  imu: {im['sample_count']} samples")
    cs = manifest["calc_skew_ns"]
    print(f"  4-cam calc_skew: mean={cs['mean']}ns max={cs['max']}ns (n={cs['n']})")
    rw = manifest["recording_window"]
    print(f"  mac duration: {rw['mac']['duration_s']:.2f}s  "
          f"board duration: {rw['board']['duration_s']}")
    print(f"\n[verify] python3 sync_verify.py --cam-log {out_dir/'cam_diag.log'} --imu-log {out_dir/'imu.log'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
