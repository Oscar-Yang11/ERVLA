#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON_BIN:-python}" "${SCRIPT_DIR}/../tools/datasets/annotate_cot_from_raw.py" "$@"
