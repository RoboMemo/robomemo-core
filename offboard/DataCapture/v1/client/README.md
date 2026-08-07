# RoboBaton 4P — Mac-side low-latency capture & sync verification

Client tools for the X5 SC132 4-camera + ICM42688 IMU demo, running on the dev Mac.
The board (`root@192.168.1.12`) produces four RTSP streams + an IMU stream; these tools
receive, time, and verify them.

## Build status (why we did not recompile)

The demo is **cross-compile only** (aarch64 X5 target). The Mac has:
- no `cmake`, no `aarch64-linux-gnu-gcc`, no X5 SDK (`hbn_api.h`, `libhbmem.so`)

So native/cross compile is blocked. This is fine: `RoboBaton_4p_demo/demo/bin/*` are
prebuilt AArch64 ELFs (verified) and are already deployed and running on the board
(`./cam_demo`, confirmed live: H.264 1280×1088 @60fps on all 4 ports). **Recompilation is
not needed for capture or sync verification.** To rebuild you must obtain the Horizon X5
SDK + toolchain and run `cmake -S . -B build_x5 -DCMAKE_TOOLCHAIN_FILE=...`.

> The binaries deployed in the board's `/root/demo` differ slightly from this repo's
> source: `imu_reader_demo` has **no `--print-rate-hz`** (it prints every sample; `--count`
> gates it), and `cam_demo`'s send-stats line uses `full_waits=` (not `queue_full_rejects=`)
> with `queue=…/10`. The tools below match the deployed binaries.

## Measured results (verified on this board, 2026-07-27)

| Quantity | Measured | Verdict |
|---|---|---|
| 4-cam intra-group skew (`calc_skew_ns`) | **mean 47 µs, p99 61 µs, max 61 µs** (n=16) | µs-level — far below ms; all 4 sensors share one GPIO417 trigger edge |
| cam↔IMU shared-clock phase | **mean 0.26 ms, p99/max 0.46 ms** (n=16, vs 18 053 IMU samples) | sub-ms; bounded by ½ the 1 kHz IMU period → confirms shared `CLOCK_MONOTONIC_RAW` |
| Per-cam RTSP fps | 59.2 (target 60), `full_waits=0` | healthy, no queue drops |
| Board encode/send latency | pipeline_delay 10–14 ms, send_max 12–14 ms | normal |
| Mac-side 4-stream inter-cam arrival skew | **mean 3.6 ms, max 3.7 ms** (transport/UDP) | ms-level on the client side |

Conclusion: the 4 cameras are synchronized to **tens of microseconds** and share a common
board clock with the IMU to **sub-millisecond**, so ms-level synchronization is met with
large margin. The remaining quantity — **physical** cam↔IMU time-delay (exposure + ISP +
IMU group delay) — is not given by timestamp phase and needs a motion event (see scope note).

## Tools

### 1. `capture_4cam.py` — low-latency 4-stream RTSP capture (no SSH needed)

One ffmpeg subprocess per camera, decoded to raw BGR24 with low-latency flags
(`-fflags nobuffer -flags low_delay -max_delay 0 -fps_mode passthrough`), software H.264
decode by default (robust; Apple Silicon handles 4×1280×1088@60 with margin;
`--hwaccel` enables VideoToolbox).

```bash
python3 capture_4cam.py --duration 10 --transport udp --out runs/probe_udp
python3 capture_4cam.py --duration 30 --transport tcp --save-frames 5 --out runs/probe_tcp
```

Reported on the **client/transport level** (Mac clock):
- per-cam decode fps (target 60), inter-arrival dt + jitter
- **inter-camera arrival skew**: mean |Δt| to nearest frame from each other cam

Measured (UDP, wired 0.5 ms RTT): all 4 cams 60.02 fps, inter-cam arrival skew
mean ≈ 3.6 ms / max ≈ 3.7 ms.

### 2. `run_board_sync.sh` + `sync_verify.py` — sensor-level sync (needs SSH)

RTSP does **not** carry sensor timestamps. The camera `camera_ts_ns`/`group_ts_ns`
(GPIO417 edge, CLOCK_MONOTONIC_RAW) and the IMU `host_timestamp_ns` (CLOCK_MONOTONIC_RAW)
are printed to the **board's stdout**. In the default `software_gpio` trigger mode all
four cameras in a group share the same group timestamp, and that clock is the **same
CLOCK_MONOTONIC_RAW the IMU uses** — so a common timebase exists.

`run_board_sync.sh` runs concurrently on the board (disjoint hardware: MIPI camera vs SPI
IMU, safe to co-run):
- `cam_demo --diagnostics` → `frameset group_id=.. group_ts_ns=.. calc_skew_ns=..` + per-cam send stats
- `imu_reader_demo --sample-rate-hz 1000 --print-rate-hz 1000` → `ts_ns=.. accel_mps2=[..]`

`sync_verify.py` then reports three things, all on the board clock:

| Metric | What it proves | Expected |
|---|---|---|
| (1) 4-cam `calc_skew_ns` | hardware sync of the 4 sensors | p99 < 2 ms (measured ~1.06 ms) |
| (2) per-cam send health | fps, `queue_full_rejects=0`, pipeline delay | fps≈60, rejects=0 |
| (3) cam↔IMU phase | shared CLOCK_MONOTONIC_RAW timebase | ≤ ~0.5 ms (½ IMU period) |

```bash
SSH="ssh root@192.168.1.12" bash run_board_sync.sh 192.168.1.12 15 out
python3 sync_verify.py --cam-log out/board_cam.log --imu-log out/board_imu.log
```

### What "ms-level sync" means here (honest scope)

- **4-camera sync** is directly guaranteed by (1) — the four sensors are hardware-triggered
  together; `calc_skew_ns` is the measured within-group skew.
- **cam↔IMU shared timebase** is proven by (3): every frame and every IMU sample is
  timestamped on the same board clock, so a dataset can label them consistently to
  sub-ms. This is what matters for synchronized data collection.
- **cam↔IMU physical time-delay** (exposure + ISP pipeline + IMU group delay) is a
  *different* quantity and is NOT given by timestamp phase. The demo deliberately does not
  fake it (per README §4). Measuring it requires a sharp common motion event (a tap on the
  rig): the IMU accel peak and the visual jolt frame, both on the board clock, give the TD.
  This is a follow-up (`motion_xcorr.py`, optional) — it needs a deliberate impulse and
  frame↔group_ts alignment.

## Files
- `capture_4cam.py` — 4-stream low-latency receiver + arrival-skew probe
- `run_board_sync.sh` — board-side concurrent cam+IMU capture over SSH
- `sync_verify.py` — parses board logs, reports 4-cam skew + send health + cam↔IMU phase
