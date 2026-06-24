#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
mkdir -p logs/phase3

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_TAG=${RUN_TAG:-seed43_47_lora_focus20_20260505}
PIDFILE=${PIDFILE:-logs/phase3/phase3_repeated_${RUN_TAG}_${STAMP}.pid}
LAUNCH_LOG=${LAUNCH_LOG:-logs/phase3/phase3_repeated_${RUN_TAG}_${STAMP}_launcher.log}

{
  echo "[$(date '+%F %T')] launching run_phase3_repeated_seeds.sh"
  echo "run_tag: $RUN_TAG"
  echo "main_log: logs/phase3/phase3_repeated_${RUN_TAG}.log"
} > "$LAUNCH_LOG"

nohup env \
  RUN_TAG="$RUN_TAG" \
  bash "$PWD/run_phase3_repeated_seeds.sh" \
  > /dev/null 2>&1 &

echo $! > "$PIDFILE"
{
  echo "pid: $(cat "$PIDFILE")"
  echo "pidfile: $PIDFILE"
  echo "launcher_log: $LAUNCH_LOG"
  echo "main_log: logs/phase3/phase3_repeated_${RUN_TAG}.log"
} | tee -a "$LAUNCH_LOG"