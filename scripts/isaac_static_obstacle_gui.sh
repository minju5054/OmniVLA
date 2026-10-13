#!/usr/bin/env bash
# Reuse the existing viewer and original recorded RGB; no inference or new trial.
set -euo pipefail
static_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${static_root}/scripts/isaac_gui_replay.sh" \
    --run "${static_root}/outputs/static-obstacle/scientific-001" \
    --config "${static_root}/research/frozen-edge-static-obstacle.json" "$@"
