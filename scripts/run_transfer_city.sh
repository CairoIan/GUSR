#!/usr/bin/env bash
set -euo pipefail
CITY=${1:?target city required}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_NAME=${RUN_NAME:-${CITY}_transfer_nonroad}
OSM_MODE=${OSM_MODE:-nonroad}
RUN_NAME="$RUN_NAME" SEED="${SEED:-192}" bash "$SCRIPT_DIR/run_stage2.sh" "$CITY" "${@:2}"
