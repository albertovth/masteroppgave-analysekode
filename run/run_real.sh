#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PY="${PY:-python3}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data}"
DATA_FILE="${DATA_FILE:-restricted_input_file.xlsx}"

REDACT_LOGS=1 \
RUN_MODE=REAL \
DATA_DIR="$DATA_DIR" \
DATA_FILE="$DATA_FILE" \
"$PY" "$SCRIPT_DIR/analyse_skript_masteroppgave11.py"
