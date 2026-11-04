#!/usr/bin/env bash
# Control-host side: copy the rental ledger to the reaper host every INTERVAL seconds (default 120). Usage: replicate.sh LEDGER HOST REMOTE_PATH [INTERVAL]
set -u
LEDGER=$1; HOST=$2; DEST=$3; INTERVAL=${4:-120}
while true; do
  if [ -f "$LEDGER" ]; then scp -q -o BatchMode=yes -o ConnectTimeout=10 "$LEDGER" "$HOST:$DEST.tmp" && ssh -o BatchMode=yes "$HOST" "mv -f '$DEST.tmp' '$DEST'" || echo "$(date -u +%FT%TZ) replicate failed" >&2; fi
  sleep "$INTERVAL"
done
