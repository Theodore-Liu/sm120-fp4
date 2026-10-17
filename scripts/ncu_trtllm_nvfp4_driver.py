"""Driver for Nsight Compute: four calls of one TensorRT-LLM NVFP4 GEMM backend at 2048 x 14336 x 4096 (profile the 4th).

    ncu --launch-skip 3 --launch-count 1 --kernel-name regex:"(?i)gemm|cutlass|nvfp4|fp4|sm120|kernel" --metrics ... python scripts/ncu_trtllm_nvfp4_driver.py cutlass
"""
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import tensorrt_llm  # noqa: E402,F401
from trtllm_nvfp4_gemm_probe import backends, lossless_operands  # noqa: E402
from sm120fp4.layouts import to_128x4  # noqa: E402

be = sys.argv[1]
M, N, K = (int(x) for x in (sys.argv[2:5] or (2048, 14336, 4096)))
fn = backends()[be]
gen = torch.Generator().manual_seed(7)
(ap_, asf, ags), (bp_, bsf, bgs) = lossless_operands(M, N, K, gen)
a = ap_.cuda(); b = bp_.cuda()
asw = to_128x4(asf).contiguous().cuda(); bsw = to_128x4(bsf).contiguous().cuda()
alpha = torch.ones(1, device="cuda")
for _ in range(4):
    o = fn(a, b, asw, bsw, alpha)
torch.cuda.synchronize()
print("done", be, (M, N, K))
