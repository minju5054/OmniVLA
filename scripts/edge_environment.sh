#!/usr/bin/env bash
set -euo pipefail
export UV_CACHE_DIR="$PWD/artifacts-local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/artifacts-local/python"
case "${1:?blackwell|official-cpu}" in
  blackwell) envdir=artifacts-local/envs/edge-blackwell; backend=cu128; torchver=2.7.1; visionver=0.22.1 ;;
  official-cpu) envdir=artifacts-local/envs/edge-official-cpu; backend=cu121; torchver=2.2.0; visionver=0.17.0 ;;
  *) exit 2 ;;
esac
test ! -e "$envdir"
uv venv --no-config --python 3.10 "$envdir"
# Official-family CPU reference reuses cached official CUDA wheels but executes
# with CUDA_VISIBLE_DEVICES=''; no old environment is modified.
uv pip install --no-config --python "$envdir/bin/python" --dry-run "torch==$torchver" "torchvision==$visionver" --index-url "https://download.pytorch.org/whl/$backend"
uv pip install --no-config --python "$envdir/bin/python" "torch==$torchver" "torchvision==$visionver" --index-url "https://download.pytorch.org/whl/$backend"
uv pip install --no-config --python "$envdir/bin/python" --torch-backend "$backend" --dry-run -r research/requirements-edge.txt
uv pip install --no-config --python "$envdir/bin/python" --torch-backend "$backend" -r research/requirements-edge.txt
uv pip check --python "$envdir/bin/python"
uv pip freeze --python "$envdir/bin/python"
