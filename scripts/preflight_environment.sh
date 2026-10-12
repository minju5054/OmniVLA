#!/usr/bin/env bash
# Isolated installs only. Run from repository root. Logs go in artifacts-local/.
set -euo pipefail
export UV_CACHE_DIR="$PWD/artifacts-local/uv-cache"
export UV_PYTHON_INSTALL_DIR="$PWD/artifacts-local/python"
case "${1:?official|blackwell|inference}" in
  official)
    test ! -e artifacts-local/envs/official
    uv venv --no-config --python 3.10 artifacts-local/envs/official
    uv pip install --no-config --python artifacts-local/envs/official/bin/python --dry-run numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    uv pip install --no-config --python artifacts-local/envs/official/bin/python numpy==1.26.4 torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
    ;;
  blackwell)
    test ! -e artifacts-local/envs/blackwell
    uv venv --no-config --python 3.10 artifacts-local/envs/blackwell
    uv pip install --no-config --python artifacts-local/envs/blackwell/bin/python --dry-run torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    uv pip install --no-config --python artifacts-local/envs/blackwell/bin/python torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
    ;;
  inference)
    uv pip install --no-config --python artifacts-local/envs/blackwell/bin/python --torch-backend cu128 --constraint research/constraints-blackwell.txt --dry-run -r research/requirements-inference.txt
    uv pip install --no-config --python artifacts-local/envs/blackwell/bin/python --torch-backend cu128 --constraint research/constraints-blackwell.txt -r research/requirements-inference.txt
    ;;
  *) exit 2 ;;
esac
