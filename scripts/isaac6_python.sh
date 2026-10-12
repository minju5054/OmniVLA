#!/usr/bin/env bash
# Process-local isolation; the selected Isaac launcher supplies its own paths.
set -euo pipefail
root="${OMNIVLA_ISAAC6_ROOT:-${HOME}/isaacsim}"
if [[ ! -r "${root}/VERSION" ]]; then
    echo "Missing Isaac VERSION file: ${root}" >&2
    exit 2
fi
version="$(<"${root}/VERSION")"
if [[ "${version}" != 6.0.1 && "${version}" != 6.0.1-* ]]; then
    echo "Expected an explicitly selected Isaac Sim 6.0.1 installation: ${root}" >&2
    exit 2
fi
exec env -u LD_LIBRARY_PATH -u PYTHONPATH -u PYTHONEXE \
    PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 "${root}/python.sh" "$@"
