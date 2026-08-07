#!/usr/bin/env bash
# Board-side sensor capture for ms-level sync verification (tested against the
# deployed /root/demo binaries).
#
# Runs the two board demos CONCURRENTLY. Both timestamp on the board's
# CLOCK_MONOTONIC_RAW, so their stdout is directly comparable:
#   - cam_demo --diagnostics          -> `frameset ... calc_skew_ns=...` + `cam%d fps=...`
#   - imu_reader_demo --count 0       -> `ts_ns=... accel_mps2=[...] gyro_rps=[...]` @ 1 kHz
#
# They touch disjoint hardware (MIPI camera vs SPI IMU), so running both at once
# is safe; only one *camera* app may run at a time.
#
# NOTE on the deployed binaries (differ slightly from repo source):
#   - imu_reader_demo has NO --print-rate-hz; it prints every sample, --count gates it.
#   - cam_demo --diag-interval-ms is accepted but the frameset block effectively emits
#     ~1/s; run longer (>=20s) for more samples. Some cam%d lines interleave (deployed
#     binary stdout quirk); sync_verify.py regexes skip the garbled ones.
#
# Usage from the Mac (SSH key auth already set up):
#   bash run_board_sync.sh 192.168.1.12 20 out
#   python3 sync_verify.py --cam-log out/board_cam.log --imu-log out/board_imu.log
set -euo pipefail

BOARD_IP="${1:-192.168.1.12}"
DURATION="${2:-20}"
OUT="${3:-out}"

mkdir -p "$OUT"
echo "[board] concurrent cam_demo --diagnostics + imu_reader_demo for ${DURATION}s on ${BOARD_IP}"

ssh -o BatchMode=yes -o ConnectTimeout=5 "root@${BOARD_IP}" 'bash -s' <<REMOTE
set +e
cd /root/demo
killall -q cam_demo imu_reader_demo sensor_demo 2>/dev/null
sleep 1
/etc/init.d/S90cam-service start 2>/dev/null || true
sleep 1
rm -f /root/demo/cam_diag.log /root/demo/imu.log
./cam_demo --diagnostics > /root/demo/cam_diag.log 2>&1 &
CAMPID=\$!
sleep 3
./imu_reader_demo --sample-rate-hz 1000 --count 0 > /root/demo/imu.log 2>&1 &
IMUPID=\$!
sleep ${DURATION}
kill -INT \$CAMPID \$IMUPID 2>/dev/null
sleep 2
kill -9 \$CAMPID \$IMUPID 2>/dev/null
echo "cam lines=\$(wc -l < /root/demo/cam_diag.log) imu lines=\$(wc -l < /root/demo/imu.log)"
# restore a plain cam_demo so RTSP stays up
nohup ./cam_demo > /root/demo/cam_demo.restore.log 2>&1 &
REMOTE

echo "[board] fetching logs"
scp -o BatchMode=yes -o ConnectTimeout=5 "root@${BOARD_IP}:/root/demo/cam_diag.log" "$OUT/board_cam.log"
scp -o BatchMode=yes -o ConnectTimeout=5 "root@${BOARD_IP}:/root/demo/imu.log" "$OUT/board_imu.log"
ssh -o BatchMode=yes "root@${BOARD_IP}" 'rm -f /root/demo/cam_diag.log /root/demo/imu.log'

echo "[done]  $OUT/board_cam.log  $OUT/board_imu.log"
echo "[next]  python3 sync_verify.py --cam-log $OUT/board_cam.log --imu-log $OUT/board_imu.log"
