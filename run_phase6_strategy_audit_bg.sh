#!/usr/bin/env bash
# Start phase-6 strategy-audit analyses in the background.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
mkdir -p logs/phase6

STAMP=${STAMP:-$(date +%Y%m%d_%H%M%S)}
RUN_TAG=${RUN_TAG:-strategy_audit_${STAMP}}
LOG_FILE=${LOG_FILE:-./logs/phase6/phase6_${RUN_TAG}.log}
PIDFILE=${PIDFILE:-./logs/phase6/phase6_${RUN_TAG}.pid}
BOOT_LOG=${BOOT_LOG:-./logs/phase6/phase6_${RUN_TAG}_nohup.log}

RUN_TAG="$RUN_TAG" LOG_FILE="$LOG_FILE" nohup bash run_phase6_strategy_audit.sh > "$BOOT_LOG" 2>&1 &
PID=$!
echo "$PID" > "$PIDFILE"

echo "started phase6 strategy-audit analyses"
echo "pid: $PID"
echo "pidfile: $PIDFILE"
echo "log: $LOG_FILE"
echo "nohup log: $BOOT_LOG"
echo "monitor: tail -f $LOG_FILE"