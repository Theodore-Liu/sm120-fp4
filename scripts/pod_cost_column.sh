#!/usr/bin/env bash
# R1's second architecture: run the cost-column and cross-library scripts on a rented SM120 box (RTX PRO 6000) with the
# same library versions as the RTX 5090 tables, writing everything under $OUT for scp back into reports/.
#
# Two venvs because the pins conflict: A = torch 2.13.0+cu130 with vLLM 0.28.0 and FlashInfer 0.7.1 (+ the DeepGEMM fork
# built from source), B = TensorRT-LLM 1.3.0rc9 (pins torch 2.9.1+cu128). FlashInfer's JIT ninja is capped at -j4 by a
# PATH shim (the 5090 host filled 40 GB with the default parallelism). Every step records its exit code; nothing aborts the
# rest.
#
#   git clone https://github.com/Theodore-Liu/sm120-fp4 /workspace/sm120-fp4   (or tar the tree over ssh)
#   bash /workspace/sm120-fp4/scripts/pod_cost_column.sh
#   scp -r <pod>:/workspace/out/. reports/<device>-cost-column-<date>/
set -u
REPO=${REPO:-/workspace/sm120-fp4}
OUT=${OUT:-/workspace/out}
VA=${VA:-/workspace/venv-a}
VB=${VB:-/workspace/venv-b}
WITH_TRTLLM=${WITH_TRTLLM:-1}
WITH_DEEPGEMM=${WITH_DEEPGEMM:-1}
mkdir -p "$OUT" /workspace/bin
exec > >(tee -a "$OUT/run.log") 2>&1
echo "START $(date -u +%FT%TZ)"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv,noheader | tee "$OUT/gpu.txt"
nproc; free -g | sed -n 2p

# ninja shim first on PATH for every venv
printf '%s\n' '#!/bin/sh' 'exec "$(dirname "$0")/../real-ninja/ninja" -j4 "$@"' > /workspace/bin/ninja; chmod +x /workspace/bin/ninja

# ---- venv A: torch 2.13 cu130, vLLM 0.28.0, FlashInfer 0.7.1
[ -x "$VA/bin/python" ] || python3 -m venv "$VA"
PA="$VA/bin/python"
"$PA" -m pip install -q --upgrade pip >/dev/null 2>&1
"$PA" -m pip install -q "torch==2.13.0" --index-url https://download.pytorch.org/whl/cu130 2>&1 | tail -1
"$PA" -m pip install -q "vllm==0.28.0" "flashinfer-python==0.7.1" "nvidia-cutlass-dsl[cu13]" ninja pytest 2>&1 | tail -1
mkdir -p /workspace/real-ninja && ln -sf "$VA/bin/ninja" /workspace/real-ninja/ninja
"$PA" - <<'EOF' | tee "$OUT/versions-a.txt"
import torch, vllm, flashinfer
print("torch", torch.__version__, "cuda", torch.version.cuda); print("vllm", vllm.__version__); print("flashinfer", flashinfer.__version__)
print("device", torch.cuda.get_device_name(0), torch.cuda.get_device_capability())
EOF
echo "SETUP_A_RC=$?"
export PYTHONPATH="$REPO:$REPO/scripts"
export PATH="/workspace/bin:$VA/bin:$PATH" FLASHINFER_NVCC_THREADS=1 VLLM_USE_V2_MODEL_RUNNER=0
cd "$REPO"
"$PA" scripts/vllm_kernel_throughput.py --out "$OUT/vllm-kernel-throughput.json" > "$OUT/vllm-kernel-throughput.log" 2>&1; echo "VLLM_THROUGHPUT_RC=$?"
"$PA" scripts/flashinfer_kernel_throughput.py --out "$OUT/flashinfer-kernel-throughput.json" > "$OUT/flashinfer-kernel-throughput.log" 2>&1; echo "FLASHINFER_THROUGHPUT_RC=$?"
"$PA" scripts/flashinfer_auto_dispatch_cost.py --m 16 --n 4096 --k 4096 > "$OUT/auto-dispatch-m16.log" 2>&1; echo "AUTO_DISPATCH_RC=$?"
mkdir -p "$OUT/xlib" "$OUT/xquant"
"$PA" scripts/nvfp4_gemm_cross_library.py --phase dump --lib flashinfer --outdir "$OUT/xlib" > "$OUT/xlib/dump-flashinfer.log" 2>&1; echo "XLIB_FI_RC=$?"
"$PA" scripts/nvfp4_quantize_cross_library.py --phase dump --lib reference --outdir "$OUT/xquant" > "$OUT/xquant/dump-reference.log" 2>&1
"$PA" scripts/nvfp4_quantize_cross_library.py --phase dump --lib vllm --outdir "$OUT/xquant" > "$OUT/xquant/dump-vllm.log" 2>&1; echo "XQUANT_VLLM_RC=$?"
"$PA" scripts/nvfp4_quantize_cross_library.py --phase dump --lib flashinfer --outdir "$OUT/xquant" > "$OUT/xquant/dump-flashinfer.log" 2>&1; echo "XQUANT_FI_RC=$?"

# ---- DeepGEMM fork (vllm-project/DeepGEMM dev) built into venv A
if [ "$WITH_DEEPGEMM" = "1" ]; then
  if [ ! -d /workspace/DeepGEMM ]; then git clone -q --recursive -b dev https://github.com/vllm-project/DeepGEMM /workspace/DeepGEMM; fi
  (cd /workspace/DeepGEMM && git rev-parse --short HEAD | tee "$OUT/deepgemm-commit.txt" && "$PA" -m pip install -q -e . --no-build-isolation 2>&1 | tail -2); echo "DEEPGEMM_BUILD_RC=$?"
  PYTHONPATH="$PYTHONPATH:/workspace/DeepGEMM/tests" "$PA" scripts/deepgemm_kernel_throughput.py --out "$OUT/deepgemm-kernel-throughput.json" > "$OUT/deepgemm-kernel-throughput.log" 2>&1; echo "DEEPGEMM_THROUGHPUT_RC=$?"
fi

# ---- venv B: TensorRT-LLM
if [ "$WITH_TRTLLM" = "1" ]; then
  [ -x "$VB/bin/python" ] || python3 -m venv "$VB"
  PB="$VB/bin/python"
  "$PB" -m pip install -q --upgrade pip >/dev/null 2>&1
  "$PB" -m pip install -q "tensorrt_llm==1.3.0rc9" --extra-index-url https://pypi.nvidia.com 2>&1 | tail -2; echo "SETUP_B_RC=$?"
  "$PB" -c "import tensorrt_llm, torch; print('trtllm', tensorrt_llm.__version__, 'torch', torch.__version__)" 2>/dev/null | tail -1 | tee "$OUT/versions-b.txt"
  "$PB" scripts/trtllm_kernel_throughput.py --out "$OUT/trtllm-kernel-throughput.json" > "$OUT/trtllm-kernel-throughput.log" 2>&1; echo "TRTLLM_THROUGHPUT_RC=$?"
  "$PB" scripts/nvfp4_gemm_cross_library.py --phase dump --lib trtllm --outdir "$OUT/xlib" > "$OUT/xlib/dump-trtllm.log" 2>&1; echo "XLIB_TRT_RC=$?"
  "$PB" scripts/nvfp4_quantize_cross_library.py --phase dump --lib trtllm --outdir "$OUT/xquant" > "$OUT/xquant/dump-trtllm.log" 2>&1; echo "XQUANT_TRT_RC=$?"
fi
"$PA" scripts/nvfp4_gemm_cross_library.py --phase compare --outdir "$OUT/xlib" --out "$OUT/xlib/cross-library.json" > "$OUT/xlib/compare.log" 2>&1; echo "XLIB_COMPARE_RC=$?"
"$PA" scripts/nvfp4_cross_library_sanity.py --outdir "$OUT/xlib" > "$OUT/xlib/sanity.log" 2>&1
"$PA" scripts/nvfp4_quantize_cross_library.py --phase compare --outdir "$OUT/xquant" --out "$OUT/xquant/quant-cross-library.json" > "$OUT/xquant/compare.log" 2>&1; echo "XQUANT_COMPARE_RC=$?"
rm -f "$OUT"/xlib/*.pt "$OUT"/xquant/*.pt   # tensors stay on the pod; only JSON and logs come back
echo "WRAPPER_RC=0"
echo "END $(date -u +%FT%TZ)"
