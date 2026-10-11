#!/usr/bin/env bash
# Owed item 2 of the grouped block-scale fix: a fresh venv with FlashInfer main (the fix worktree) installed editable, so
# tests/grouped_mm runs against main's own code instead of the 0.7.1 wheel with an overlay.
set -euo pipefail
WT=/mnt/c/Users/jingz/oss/flashinfer-grouped-sfa
V=~/fimain
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
[ -x $V/bin/python ] || ~/.local/bin/uv venv -p 3.12 $V
~/.local/bin/uv pip install -p $V/bin/python torch --index-url https://download.pytorch.org/whl/cu130
~/.local/bin/uv pip install -p $V/bin/python "setuptools>=77" wheel ninja pytest numpy
~/.local/bin/uv pip install -p $V/bin/python "nvidia-cudnn-cu13>=9.12" "nvidia-cudnn-frontend>=1.13" || true
cd $WT
~/.local/bin/uv pip install -p $V/bin/python --no-build-isolation -e . -v
$V/bin/python -c "import flashinfer, torch; print('flashinfer', flashinfer.__version__, flashinfer.__file__, 'torch', torch.__version__)"
echo BUILD_RC=0
