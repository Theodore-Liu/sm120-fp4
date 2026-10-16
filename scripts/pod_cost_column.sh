#!/usr/bin/env bash
# R1's second architecture: run the cost-column and cross-library scripts on a rented SM120 box (RTX PRO 6000) with the
# same library versions as the RTX 5090 tables, writing everything under $OUT for scp back into reports/.
#
# Three venvs because the pins conflict: A = vLLM 0.28.0 (its own torch 2.13 and FlashInfer 0.6.16.post3 pins) plus the
# DeepGEMM fork built from source; C = torch 2.13.0+cu130 with FlashInfer 0.7.1 (the 5090 FlashInfer table's version);
# B = TensorRT-LLM 1.3.0rc9 (pins torch 2.9.1+cu128). The system CUDA 13.2 toolkit comes from NVIDIA's apt repository, as
# pod_conformance.sh does: the image's 12.8 toolkit makes FlashInfer refuse b12x and fail its JIT, DeepGEMM's build needs
# nvcc 13, and TensorRT-LLM dlopens libcublasLt.so.13 and libmpi.so.40. FlashInfer's JIT ninja is capped at -j4 by a PATH
# shim (the 5090 host filled 40 GB with the default parallelism). Every step records its exit code; nothing aborts the rest.
#
#   bash /workspace/sm120-fp4/scripts/pod_cost_column.sh      (started by mlsys scripts/pod/cost-column.sh via pod-launch.sh)
#   scp -r <pod>:/workspace/out/. reports/<device>-cost-column-<date>/
set -u
REPO=${REPO:-/workspace/sm120-fp4}
OUT=${OUT:-/workspace/out}
VA=${VA:-/workspace/venv-a}
VB=${VB:-/workspace/venv-b}
VC=${VC:-/workspace/venv-c}
WITH_TRTLLM=${WITH_TRTLLM:-1}
WITH_DEEPGEMM=${WITH_DEEPGEMM:-1}
CUDA_APT_VER=${CUDA_APT_VER:-13-2}
mkdir -p "$OUT" /workspace/bin
exec > >(tee -a "$OUT/run.log") 2>&1
echo "START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | tee "$OUT/gpu.txt"
nproc; free -g | sed -n 2p

# ---- system CUDA 13.2 toolkit (compiler and headers from one release) and OpenMPI
if [ ! -x "/usr/local/cuda-${CUDA_APT_VER/-/.}/bin/nvcc" ]; then
  apt-get update -qq > "$OUT/apt.log" 2>&1
  # libdw-dev: DeepGEMM's deep_jit includes elfutils/libdwfl.h
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "cuda-toolkit-$CUDA_APT_VER" libopenmpi-dev openmpi-bin libdw-dev >> "$OUT/apt.log" 2>&1; echo "APT_RC=$?"
fi
export CUDA_HOME="/usr/local/cuda-${CUDA_APT_VER/-/.}"
export PATH="/workspace/bin:$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
echo "CUDA_HOME=$CUDA_HOME nvcc=$(command -v nvcc || echo none) $(nvcc --version 2>/dev/null | tail -1)"
ls "$CUDA_HOME/lib64/libcublasLt.so.13" "$CUDA_HOME/targets/x86_64-linux/lib/libcublasLt.so.13" 2>/dev/null | head -1

# ninja shim first on PATH for every venv
printf '%s\n' '#!/bin/sh' 'exec /workspace/real-ninja/ninja -j4 "$@"' > /workspace/bin/ninja; chmod +x /workspace/bin/ninja
export FLASHINFER_NVCC_THREADS=1 VLLM_USE_V2_MODEL_RUNNER=0 PYTHONPATH="$REPO:$REPO/scripts"

# ---- venv A: vLLM 0.28.0 (brings its own torch 2.13 and FlashInfer 0.6.16.post3)
[ -x "$VA/bin/python" ] || python3 -m venv "$VA"
PA="$VA/bin/python"
"$PA" -m pip install -q --upgrade pip >/dev/null 2>&1
"$PA" -m pip install -q "vllm==0.28.0" --extra-index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1; echo "PIP_A_RC=$?"
"$PA" -m pip install -q ninja pytest 2>&1 | tail -1
"$PA" - <<'EOF' | tee "$OUT/versions-a.txt"
import torch, vllm
print("torch", torch.__version__, "cuda", torch.version.cuda); print("vllm", vllm.__version__)
try:
    import flashinfer; print("flashinfer (vllm's pin)", flashinfer.__version__)
except Exception as e: print("flashinfer absent in venv A:", e)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
EOF
echo "SETUP_A_RC=$?"

# ---- venv C: torch 2.13 cu130 + FlashInfer 0.7.1 + cuDNN >= 9.23 (mm_bf16_fp4's cudnn backend needs 9.23.1)
[ -x "$VC/bin/python" ] || python3 -m venv "$VC"
PC="$VC/bin/python"
"$PC" -m pip install -q --upgrade pip >/dev/null 2>&1
"$PC" -m pip install -q "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
"$PC" -m pip install -q "flashinfer-python==0.7.1" "nvidia-cutlass-dsl[cu13]" "nvidia-cudnn-cu13>=9.23" ninja 2>&1 | tail -1; echo "PIP_C_RC=$?"
mkdir -p /workspace/real-ninja && ln -sf "$VC/bin/ninja" /workspace/real-ninja/ninja
"$PC" -c "import torch, flashinfer; print('venv C torch', torch.__version__, 'flashinfer', flashinfer.__version__, 'cudnn', torch.backends.cudnn.version())" 2>&1 | tail -1 | tee "$OUT/versions-c.txt"
echo "SETUP_C_RC=$?"

cd "$REPO"
"$PA" scripts/vllm_kernel_throughput.py --out "$OUT/vllm-kernel-throughput.json" > "$OUT/vllm-kernel-throughput.log" 2>&1; echo "VLLM_THROUGHPUT_RC=$?"
"$PC" scripts/flashinfer_kernel_throughput.py --out "$OUT/flashinfer-kernel-throughput.json" > "$OUT/flashinfer-kernel-throughput.log" 2>&1; echo "FLASHINFER_THROUGHPUT_RC=$?"
"$PC" scripts/flashinfer_auto_dispatch_cost.py --m 16 --n 4096 --k 4096 > "$OUT/auto-dispatch-m16.log" 2>&1; echo "AUTO_DISPATCH_RC=$?"
mkdir -p "$OUT/xlib" "$OUT/xquant"
"$PC" scripts/nvfp4_gemm_cross_library.py --phase dump --lib flashinfer --outdir "$OUT/xlib" > "$OUT/xlib/dump-flashinfer.log" 2>&1; echo "XLIB_FI_RC=$?"
"$PC" scripts/nvfp4_quantize_cross_library.py --phase dump --lib reference --outdir "$OUT/xquant" > "$OUT/xquant/dump-reference.log" 2>&1
"$PA" scripts/nvfp4_quantize_cross_library.py --phase dump --lib vllm --outdir "$OUT/xquant" > "$OUT/xquant/dump-vllm.log" 2>&1; echo "XQUANT_VLLM_RC=$?"
"$PC" scripts/nvfp4_quantize_cross_library.py --phase dump --lib flashinfer --outdir "$OUT/xquant" > "$OUT/xquant/dump-flashinfer.log" 2>&1; echo "XQUANT_FI_RC=$?"

# ---- DeepGEMM fork (vllm-project/DeepGEMM dev) built into venv A with the CUDA 13.2 toolkit
if [ "$WITH_DEEPGEMM" = "1" ]; then
  if [ ! -d /workspace/DeepGEMM ]; then git clone -q --recursive -b dev https://github.com/vllm-project/DeepGEMM /workspace/DeepGEMM; fi
  # pip's editable install skips develop.sh's step that links the CUTLASS headers into deep_gemm/include; without them every
  # JIT compile fails ("command failed with exit code 1" from deep_jit)
  (cd /workspace/DeepGEMM && ln -sf /workspace/DeepGEMM/third-party/cutlass/include/cutlass deep_gemm/include/cutlass && ln -sf /workspace/DeepGEMM/third-party/cutlass/include/cute deep_gemm/include/cute)
  (cd /workspace/DeepGEMM && git rev-parse --short HEAD | tee "$OUT/deepgemm-commit.txt" && "$PA" -m pip install -e . --no-build-isolation > "$OUT/deepgemm-build.log" 2>&1); rc=${PIPESTATUS[0]}; echo "DEEPGEMM_BUILD_RC=$rc"; tail -3 "$OUT/deepgemm-build.log"
  PYTHONPATH="$PYTHONPATH:/workspace/DeepGEMM/tests" "$PA" scripts/deepgemm_kernel_throughput.py --out "$OUT/deepgemm-kernel-throughput.json" > "$OUT/deepgemm-kernel-throughput.log" 2>&1; echo "DEEPGEMM_THROUGHPUT_RC=$?"
fi

# ---- venv B: TensorRT-LLM
if [ "$WITH_TRTLLM" = "1" ]; then
  [ -x "$VB/bin/python" ] || python3 -m venv "$VB"
  PB="$VB/bin/python"
  "$PB" -m pip install -q --upgrade pip >/dev/null 2>&1
  "$PB" -m pip install -q "tensorrt_llm==1.3.0rc9" --extra-index-url https://pypi.nvidia.com 2>&1 | tail -2; echo "SETUP_B_RC=$?"
  "$PB" -c "import tensorrt_llm, torch; print('trtllm', tensorrt_llm.__version__, 'torch', torch.__version__)" 2>&1 | grep -v -i "warning\|parakeet" | tail -1 | tee "$OUT/versions-b.txt"
  "$PB" scripts/trtllm_kernel_throughput.py --out "$OUT/trtllm-kernel-throughput.json" > "$OUT/trtllm-kernel-throughput.log" 2>&1; echo "TRTLLM_THROUGHPUT_RC=$?"
  "$PB" scripts/nvfp4_gemm_cross_library.py --phase dump --lib trtllm --outdir "$OUT/xlib" > "$OUT/xlib/dump-trtllm.log" 2>&1; echo "XLIB_TRT_RC=$?"
  "$PB" scripts/nvfp4_quantize_cross_library.py --phase dump --lib trtllm --outdir "$OUT/xquant" > "$OUT/xquant/dump-trtllm.log" 2>&1; echo "XQUANT_TRT_RC=$?"
fi
"$PC" scripts/nvfp4_gemm_cross_library.py --phase compare --outdir "$OUT/xlib" --out "$OUT/xlib/cross-library.json" > "$OUT/xlib/compare.log" 2>&1; echo "XLIB_COMPARE_RC=$?"
"$PC" scripts/nvfp4_cross_library_sanity.py --outdir "$OUT/xlib" > "$OUT/xlib/sanity.log" 2>&1
"$PC" scripts/nvfp4_quantize_cross_library.py --phase compare --outdir "$OUT/xquant" --out "$OUT/xquant/quant-cross-library.json" > "$OUT/xquant/compare.log" 2>&1; echo "XQUANT_COMPARE_RC=$?"
rm -f "$OUT"/xlib/*.pt "$OUT"/xquant/*.pt   # tensors stay on the pod; only JSON and logs come back
echo "WRAPPER_RC=0"
echo "END $(date -u +%FT%TZ)"
