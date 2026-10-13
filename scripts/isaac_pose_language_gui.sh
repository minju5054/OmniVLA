#!/usr/bin/env bash
# Replay the four recorded Modality-8 trials; no model inference.
set -euo pipefail
language_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${language_root}/scripts/isaac_gui_replay.sh" \
  --config "${language_root}/research/frozen-edge-static-obstacle.json" \
  --run "${language_root}/outputs/pose-language-obstacle/navigation-001" \
  --language-protocol "${language_root}/research/frozen-pose-language-obstacle.json" "$@"
