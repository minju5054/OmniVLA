#!/usr/bin/env bash
# Replay saved results in Isaac GUI; defaults to edge-final/scientific-001.
# Forward --run/--config to select another compatible experiment explicitly.
set -euo pipefail
replay_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
replay_output="${replay_root}/artifacts-local/gui-replay/replay-$(date -u +%Y%m%dT%H%M%S.%NZ)-$$"
exec bash "${replay_root}/scripts/isaac6_python.sh" \
    "${replay_root}/research/isaac_gui_replay.py" \
    --output "${replay_output}" "$@"
