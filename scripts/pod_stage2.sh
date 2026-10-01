#!/usr/bin/env bash
# Run the stage-2 MoE layer benches on a fresh cloud GPU box (synthetic weights of the Qwen3-30B-A3B shape), writing
# everything under $OUT, so the RTX 5090 numbers in docs/stage2-design.md can be set beside another SM120 part.
#
# The environment is scripts/pod_conformance.sh's (same pins, same CUDA 13.2 apt toolkit, same three traps avoided);
# vLLM is not installed here, so the Marlin column is absent and the layer is compared only against itself and the
# stream read. The real-weight layer (scripts/real_ckpt_layer.py) needs the checkpoint shards and is not run here.
#
#   tar -C <repo> -cf - . | ssh <pod> 'mkdir -p /workspace/sm120-fp4 && tar -C /workspace/sm120-fp4 -xf -'
#   ssh <pod> 'bash /workspace/sm120-fp4/scripts/pod_stage2.sh'
#   scp -r <pod>:/workspace/out/. reports/<device>/
set -u
REPO=${REPO:-/workspace/sm120-fp4}
OUT=${OUT:-/workspace/out}
VENV=${VENV:-/workspace/venv}
TAG=${TAG:-$(date -u +%Y-%m-%d)}
TORCH_INDEX=${TORCH_INDEX:-https://download.pytorch.org/whl/cu130}
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
"$PY" - <<'EOF' | tee "$OUT/versions.txt"
import torch
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
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
cd "$REPO" || exit 7
export PYTHONPATH="$REPO" MAX_JOBS=${MAX_JOBS:-8}
DEV=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '-' | tr -cd 'A-Za-z0-9-' | tr 'A-Z' 'a-z')

run() {  # name, then the command
  local name=$1; shift
  echo "== $name"
  "$@"; echo "${name}_RC=$?"
}
run floor      "$PY" scripts/micro_floor.py --out "$OUT/micro-floor-$DEV-$TAG.json"
run layer      "$PY" scripts/moe_layer.py --out "$OUT/moe-layer-$DEV-$TAG.json"
run groups     "$PY" scripts/fc2_groups.py --out "$OUT/fc2-groups-$DEV-$TAG.json"
run groupswarm "$PY" scripts/fc2_groups.py --act-warm --out "$OUT/fc2-groups-actwarm-$DEV-$TAG.json"
run occupancy  "$PY" scripts/fc2_occupancy.py --out "$OUT/fc2-occupancy-$DEV-$TAG.json"
run occwarm    "$PY" scripts/fc2_occupancy.py --act-warm --out "$OUT/fc2-occupancy-actwarm-$DEV-$TAG.json"
run split      "$PY" scripts/fc2_split.py --out "$OUT/fc2-split-$DEV-$TAG.json"
run splitwarm  "$PY" scripts/fc2_split.py --act-warm --out "$OUT/fc2-split-actwarm-$DEV-$TAG.json"
run chain      "$PY" scripts/fc2_chain.py --out "$OUT/fc2-chain-$DEV-$TAG.json"
run chainwarm  "$PY" scripts/fc2_chain.py --act-warm --out "$OUT/fc2-chain-actwarm-$DEV-$TAG.json"
run breakdown  "$PY" scripts/moe_layer_breakdown.py --out "$OUT/moe-layer-breakdown-$DEV-$TAG.json"
echo "END $(date -u +%FT%TZ)"
