#!/usr/bin/env bash
# Owed item 2 of the grouped block-scale fix: a venv with FlashInfer main + the fix commit installed editable, so
# tests/grouped_mm runs against main's own code instead of the 0.7.1 wheel with an overlay.
# - Source: a clone of the local fix branch onto WSL's ext4 disk (~/fi-src), submodules from the local module store.
#   Building from the /mnt/c worktree took the Ubuntu distro down (setuptools walks every file over drvfs), as did
#   pre-commit there and two 3 GB torch downloads (2026-10-11).
# - venv: a local copy of ~/fi071/.venv (torch 2.9.1+cu128, cuDNN 9.27, cudnn-frontend 1.31, cutlass-dsl 4.8, tvm-ffi
#   0.1.14 satisfy main's requirements.txt); FLASHINFER_BUILD_NO_PIP=1 stops the build hook's own pip downloads.
set -euo pipefail
WIN=/mnt/c/Users/jingz/oss/flashinfer
BR=fix/grouped-block-scale-per-group-sfa
S=~/fi-src
V=~/fimain
export PATH=/usr/local/cuda-13.0/bin:$PATH CUDA_HOME=/usr/local/cuda-13.0
if [ ! -d $S/.git ]; then
  git clone --depth 2 --branch $BR file://$WIN $S
  for m in cutlass spdlog cccl; do
    sha=$(git -C $S ls-tree HEAD 3rdparty/$m | awk '{print $3}')
    rm -rf $S/3rdparty/$m
    git clone --no-checkout file://$WIN/.git/worktrees/flashinfer-grouped-sfa/modules/3rdparty/$m $S/3rdparty/$m
    git -C $S/3rdparty/$m checkout -q $sha
    echo "submodule $m at $sha"
  done
fi
git -C $S fetch -q file://$WIN $BR && git -C $S reset -q --hard FETCH_HEAD
echo "source at $(git -C $S log --oneline -1)"
if [ ! -f $V/.cloned-from-fi071 ]; then
  rm -rf $V; cp -a ~/fi071/.venv $V
  sed -i "s#$HOME/fi071/.venv#$V#g" $V/bin/activate $V/pyvenv.cfg 2>/dev/null || true
  touch $V/.cloned-from-fi071
fi
~/.local/bin/uv pip uninstall -p $V/bin/python flashinfer-python || true
cd $S
FLASHINFER_BUILD_NO_PIP=1 ~/.local/bin/uv pip install -p $V/bin/python --no-build-isolation --no-deps -e .
cd /
$V/bin/python -c "import flashinfer, torch; print('flashinfer', flashinfer.__version__, flashinfer.__file__, 'torch', torch.__version__)"
echo BUILD_RC=0
