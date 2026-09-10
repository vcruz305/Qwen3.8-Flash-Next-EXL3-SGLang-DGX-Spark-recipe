#!/bin/bash
# Zero-OOM preventive guard for GB10/UMA serving hosts (no separate VRAM pool
# on this hardware class -- /proc/meminfo MemAvailable is the only real
# signal). Polls MemAvailable and kills the owned server PID (from a pidfile)
# if it drops below a trip threshold set ABOVE the hard floor, with margin
# for allocation bursts (CUDA graph capture, KV growth) and SIGTERM/SIGKILL
# latency. Pattern verified working across multiple guard-trip/retune cycles
# on sglang-exl3 serving, 2026-09-06.
#
# Usage:
#   FLOOR_GIB=20 GUARD_GIB=22 PIDFILE=/path/to/server.pid \
#     LOGFILE=/path/to/guard.log ./memory-guard.sh
#
# Sizing GUARD_GIB -- compute BOTH margins before picking a number:
#   (a) floor-side margin: GUARD_GIB must be meaningfully above FLOOR_GIB
#       (>=1-2 GiB) so SIGTERM/SIGKILL latency doesn't let MemAvailable cross
#       the real floor before the process actually dies.
#   (b) headroom-side margin: GUARD_GIB must be BELOW your computed expected
#       steady-state headroom (ceiling - peak_load_estimate), or the guard
#       trips the instant a healthy load finishes, not just on anomalies.
# CUDA graph capture at higher --cuda-graph-max-bs causes bigger transient
# memory dips than steady-state serving or bs=1 capture -- re-verify
# empirically before raising batch size against a guard tuned for bs=1.

set -u
FLOOR_GIB=${FLOOR_GIB:-20}
GUARD_GIB=${GUARD_GIB:-22}
POLL_INTERVAL_S=${POLL_INTERVAL_S:-1}
# KILL_MODE=immediate (default): straight SIGKILL, no grace period. Verified
# necessary 2026-09-06 -- a CUDA graph capture burst dropped MemAvailable
# ~2.5 GiB in a single 1s poll tick. A graceful SIGTERM-then-wait-3s-then-
# SIGKILL sequence gives a fast burst 3 full seconds to blow through the
# hard floor before the process actually dies -- that grace period is itself
# a safety-margin risk on this hardware class, not a courtesy worth keeping.
# Set KILL_MODE=graceful only for a known slow-leak workload where SIGTERM
# cleanup matters more than kill latency.
KILL_MODE=${KILL_MODE:-immediate}
PIDFILE=${PIDFILE:?set PIDFILE to the server's pid file path}
LOGFILE=${LOGFILE:?set LOGFILE to a writable log path}

FLOOR_KB=$((FLOOR_GIB * 1024 * 1024))
GUARD_KB=$((GUARD_GIB * 1024 * 1024))

echo "$(date -u +%FT%TZ) GUARD START floor=${FLOOR_KB}kB guard=${GUARD_KB}kB poll=${POLL_INTERVAL_S}s kill_mode=${KILL_MODE}" >> "$LOGFILE"

while true; do
  AVAIL_KB=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
  TS=$(date -u +%FT%TZ)
  if [ "$AVAIL_KB" -lt "$GUARD_KB" ]; then
    echo "$TS GUARD TRIP avail=${AVAIL_KB}kB < guard=${GUARD_KB}kB -- killing owned server ($KILL_MODE)" >> "$LOGFILE"
    if [ -f "$PIDFILE" ]; then
      PID=$(cat "$PIDFILE" 2>/dev/null)
      if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
        if [ "$KILL_MODE" = "graceful" ]; then
          kill -15 "$PID" 2>/dev/null || true
          sleep 3
          kill -9 "$PID" 2>/dev/null || true
        else
          kill -9 "$PID" 2>/dev/null || true
        fi
        echo "$TS GUARD KILLED PID=$PID" >> "$LOGFILE"
      fi
    fi
    echo "$TS GUARD EXIT after trip" >> "$LOGFILE"
    exit 1
  fi
  echo "$TS avail=${AVAIL_KB}kB OK" >> "$LOGFILE"
  sleep "$POLL_INTERVAL_S"
done
