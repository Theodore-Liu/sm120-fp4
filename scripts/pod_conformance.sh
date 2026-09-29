#!/usr/bin/env bash
# Run the conformance suite, the tactic probe and the tests on a fresh cloud GPU box, writing everything under $OUT.
#
# Written against a Runpod "runpod/pytorch" Ubuntu 24.04 image (python3.12, sshd started by the platform), but nothing
# below is Runpod-specific: any Linux box with an SM12x GPU, a CUDA 13 driver and python3.12 will do. The CUDA toolkit
# comes from pip (nvidia-cuda-nvcc and friends), which is how FlashInfer's JIT finds nvcc on the development machine too,
# so the host image's CUDA version does not matter.
#
#   tar -C <repo> -cf - . | ssh <pod> 'mkdir -p /workspace/sm120-fp4 && tar -C /workspace/sm120-fp4 -xf -'
#   ssh <pod> 'bash /workspace/sm120-fp4/scripts/pod_conformance.sh'
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
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | tee "$OUT/gpu.txt"

if [ ! -x "$VENV/bin/python" ]; then
  python3 -m venv "$VENV" || { echo "SETUP_RC=venv-failed"; exit 5; }
fi
PY="$VENV/bin/python"
export PATH="$VENV/bin:$PATH"   # FlashInfer's JIT runs `ninja` by name; pip installs it here
"$PY" -m pip install -q --upgrade pip >/dev/null 2>&1
# Pinned to the versions the RTX 5090 report was produced with, so the two reports differ only in the machine.
"$PY" -m pip install -q "torch==2.13.0" --index-url "$TORCH_INDEX" 2>&1 | tail -2
"$PY" -m pip install -q "flashinfer-python==0.6.16.post3" "nvidia-cutlass-dsl[cu13]==4.6.2" \
  "nvidia-cuda-nvcc==13.3.73" "nvidia-cuda-crt==13.3.73" "nvidia-cuda-cccl==13.3.3.4.1" "nvidia-cuda-nvrtc==13.0.88" \
  "nvidia-cuda-runtime==13.0.96" "nvidia-cuda-nvdisasm==13.3.73" "cuda-python==13.3.1" pytest 2>&1 | tail -2
"$PY" - <<'EOF' | tee "$OUT/versions.txt"
import torch, flashinfer
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("flashinfer", flashinfer.__version__)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
EOF
[ $? -eq 0 ] || { echo "SETUP_RC=import-failed"; exit 6; }

# FlashInfer's JIT compiles with the toolkit in CUDA_HOME, else `which nvcc`, else /usr/local/cuda. Two traps on cloud images:
# a CUDA 12.8 toolkit at /usr/local/cuda makes it report "SM 12.x requires CUDA >= 12.9" and then fail with the misleading
# "FlashInfer requires GPUs with sm75 or higher"; and the pip nvcc (13.3) paired with torch's pip runtime headers (13.0) is
# rejected by CCCL ("CUDA compiler and CUDA toolkit headers are incompatible"). So install the same system toolkit the
# RTX 5090 report used, CUDA 13.2 from NVIDIA's apt repository, where compiler and headers come from one release.
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
echo "== conformance"
"$PY" -m sm120fp4.cli conformance --out "$OUT/conformance-$TAG.json"; echo "CONFORMANCE_RC=$?"
echo "== tactics"
"$PY" scripts/probe_tactics.py --out "$OUT/tactics-$TAG.json"; echo "TACTICS_RC=$?"
echo "== pytest"
"$PY" -m pytest -q tests 2>&1 | tail -5 | tee "$OUT/pytest-tail.txt"; echo "PYTEST_RC=${PIPESTATUS[0]}"
echo "DONE $(date -u +%FT%TZ)"
