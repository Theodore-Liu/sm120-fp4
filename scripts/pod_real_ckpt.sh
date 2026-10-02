#!/usr/bin/env bash
# Run the real-weights MoE layer table (scripts/real_ckpt_layer.py) on a fresh cloud GPU box, beside vLLM's Marlin
# W4A16 MoE, writing everything under $OUT. This is the stage-2 gate's "reproduced on an RTX PRO 6000" clause for the
# real-weights table; scripts/pod_stage2.sh covers the single-kernel benches.
#
# The environment is scripts/pod_stage2.sh's (same pins, same CUDA apt toolkit, same traps avoided) plus vLLM for the
# Marlin column and the two checkpoint shards real_ckpt_layer.py reads (it opens them with local_files_only=True, so
# they are downloaded here first). If vLLM cannot be installed against the pinned torch the run continues without the
# Marlin column and says so in VLLM_RC.
#
#   tar -C <repo> -cf - . | ssh <pod> 'mkdir -p /workspace/sm120-fp4 && tar -C /workspace/sm120-fp4 -xf -'
#   ssh <pod> 'bash /workspace/sm120-fp4/scripts/pod_real_ckpt.sh'
#   scp -r <pod>:/workspace/out/. reports/<device>/
set -u
REPO=${REPO:-/workspace/sm120-fp4}
OUT=${OUT:-/workspace/out}
VENV=${VENV:-/workspace/venv}
TAG=${TAG:-$(date -u +%Y-%m-%d)}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
VLLM_VERSION=${VLLM_VERSION:-0.28.0}
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap,clocks.max.sm,clocks.max.mem --format=csv,noheader | tee "$OUT/gpu.txt"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || { echo "SETUP_RC=venv-failed"; exit 5; }
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"
"$PY" -m pip install -q --upgrade pip >/dev/null 2>&1
"$PY" -m pip install -q "torch==2.13.0" --index-url "$TORCH_INDEX" 2>&1 | tail -2
"$PY" -m pip install -q ninja "nvidia-cuda-nvcc==13.3.73" "nvidia-cuda-crt==13.3.73" "nvidia-cuda-cccl==13.3.3.4.1" \
  "nvidia-cuda-nvrtc==13.0.88" "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-nvdisasm==13.3.73" 2>&1 | tail -2
"$PY" -m pip install -q "huggingface_hub" "safetensors" 2>&1 | tail -1
# vLLM for the Marlin column; its torch pin must agree with the installed torch, so install without touching torch and
# verify the import afterwards. A failure here leaves the Marlin column absent, not the run.
"$PY" -m pip install -q "vllm==$VLLM_VERSION" --no-deps 2>&1 | tail -2
"$PY" -m pip install -q $("$PY" -m pip show vllm 2>/dev/null | sed -n 's/^Requires: //p' | tr ',' '\n' | grep -viE '^ *(torch|torchvision|torchaudio|xformers|flashinfer.*)$' | tr '\n' ' ') 2>&1 | tail -2
"$PY" - <<'EOF' | tee "$OUT/versions.txt"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
try:
    import vllm
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe  # noqa: F401
    print("vllm", vllm.__version__, "marlin import ok")
    print("VLLM_RC=0")
except Exception as e:  # noqa: BLE001
    print("vllm import failed:", type(e).__name__, str(e)[:300])
    print("VLLM_RC=1")
EOF
[ $? -eq 0 ] || { echo "SETUP_RC=import-failed"; exit 6; }

CUDA_APT_VER=${CUDA_APT_VER:-13-2}
if [ -z "${CUDA_HOME:-}" ]; then
  if [ ! -x "/usr/local/cuda-${CUDA_APT_VER/-/.}/bin/nvcc" ]; then
    apt-get update -qq >/dev/null 2>&1
    DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-$CUDA_APT_VER" >/dev/null 2>&1 || { echo "SETUP_RC=cuda-toolkit-install-failed"; exit 8; }
  fi
  export CUDA_HOME="/usr/local/cuda-${CUDA_APT_VER/-/.}" PATH="/usr/local/cuda-${CUDA_APT_VER/-/.}/bin:$PATH"
fi
echo "CUDA_HOME=${CUDA_HOME:-unset} nvcc=$(command -v nvcc || echo none)"

# the two shards real_ckpt_layer.py opens (local_files_only): the FP4 checkpoint's first shard and the bf16 original's
"$PY" - <<'EOF'
from huggingface_hub import hf_hub_download
for repo, fname in (("nvidia/Qwen3-30B-A3B-NVFP4", "model-00001-of-00004.safetensors"),
                    ("Qwen/Qwen3-30B-A3B", "model-00001-of-00016.safetensors")):
    print("downloaded", hf_hub_download(repo, fname))
EOF
[ $? -eq 0 ] || { echo "SETUP_RC=download-failed"; exit 9; }

cd "$REPO" || exit 7
export PYTHONPATH="$REPO" MAX_JOBS=${MAX_JOBS:-8}
DEV=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -cd 'A-Za-z0-9-' | tr 'A-Z' 'a-z')
echo "== real_ckpt_layer"
"$PY" scripts/real_ckpt_layer.py --fc1-sweep --out "$OUT/real-ckpt-layer0-fc1sweep-$DEV-$TAG.json"; echo "LAYER_RC=$?"
echo "END $(date -u +%FT%TZ)"
