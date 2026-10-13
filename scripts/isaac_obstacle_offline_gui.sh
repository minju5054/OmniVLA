#!/usr/bin/env bash
# Stationary paired predictions only; preserve the existing collision launcher.
set -euo pipefail
offline_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${offline_root}/scripts/isaac_gui_replay.sh" \
  --config "${offline_root}/research/frozen-edge-static-obstacle.json" \
  --run "${offline_root}/outputs/edge-obstacle-offline/paired-001" \
  --offline-protocol "${offline_root}/research/frozen-edge-obstacle-offline.json" "$@"
