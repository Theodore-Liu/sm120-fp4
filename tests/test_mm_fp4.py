"""GEMM conformance: FlashInfer `mm_fp4` on every backend it offers, against the reference fp32 GEMM on this device.

Failure classes caught: a backend that returns all zeros or garbage without raising (FlashInfer #2577 on SM120: cuDNN
raised, CUTLASS "silently returns all zeros", TensorRT-LLM rejected the capability); and CUDA-graph replay corruption
(FlashInfer #4841, DeepGEMM #444): the captured GEMM is replayed many times and every replay is checksummed.
"""
import pytest
import torch

from sm120fp4 import quantize_nvfp4, reference_gemm_nvfp4, to_128x4

BACKENDS = ("cutlass", "cudnn", "trtllm", "b12x", "cute-dsl", "auto")


def _operands(m, n, k, device):
    torch.manual_seed(3)
    a = torch.randn(m, k, device=device).to(torch.bfloat16)
    b = torch.randn(n, k, device=device).to(torch.bfloat16)
    a_q, a_sf, a_gs = quantize_nvfp4(a.float())
    b_q, b_sf, b_gs = quantize_nvfp4(b.float())
    ref = reference_gemm_nvfp4(a_q, a_sf, a_gs, b_q, b_sf, b_gs)
    # FlashInfer's mm_fp4 wants swizzled scales (as its quantizer emits them) and alpha = 1 / (a_gs * b_gs)
    a_sf_sw = to_128x4(a_sf).view(torch.float8_e4m3fn)
    b_sf_sw = to_128x4(b_sf).view(torch.float8_e4m3fn)
    alpha = (1.0 / (a_gs * b_gs)).reshape(1)
    return a_q, a_sf_sw, b_q, b_sf_sw, alpha, ref


def _run(fi, backend, a_q, a_sf, b_q, b_sf, alpha):
    return fi.mm_fp4(a_q, b_q.t(), a_sf, b_sf, alpha, torch.bfloat16, None, 16, False, backend)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("m,n,k", [(128, 256, 512), (16, 4096, 4096), (1, 1024, 2048), (512, 512, 1024)])
def test_mm_fp4_matches_reference(device, fi, backend, m, n, k):
    a_q, a_sf, b_q, b_sf, alpha, ref = _operands(m, n, k, device)
    try:
        out = _run(fi, backend, a_q, a_sf, b_q, b_sf, alpha)
    except Exception as e:  # noqa: BLE001 - the report records the backend as unavailable, which is a finding
        pytest.skip(f"{backend} backend unavailable on this device: {type(e).__name__}: {str(e)[:160]}")
    torch.cuda.synchronize()
    out = out.float()
    assert out.shape == ref.shape
    assert not (out == 0).all(), f"{backend}: all-zero output (the FlashInfer #2577 failure class)"
    err = (out - ref).abs().max().item()
    scale = ref.abs().max().item() + 1e-6
    assert err / scale < 2e-2, f"{backend}: max abs error {err:.4g} on outputs up to {scale:.4g}"


@pytest.mark.parametrize("backend", BACKENDS)
def test_mm_fp4_graph_replay_is_stable(device, fi, backend):
    m, n, k = 64, 2048, 2048
    a_q, a_sf, b_q, b_sf, alpha, ref = _operands(m, n, k, device)
    try:
        eager = _run(fi, backend, a_q, a_sf, b_q, b_sf, alpha).clone()
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"{backend} backend unavailable: {type(e).__name__}")
    out = torch.empty_like(eager)
    s = torch.cuda.Stream()
    with torch.cuda.stream(s):
        for _ in range(2):  # warm up on the side stream before capture
            _run(fi, backend, a_q, a_sf, b_q, b_sf, alpha)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out.copy_(_run(fi, backend, a_q, a_sf, b_q, b_sf, alpha))
    bad = 0
    for i in range(200):
        g.replay()
        torch.cuda.synchronize()
        if not torch.equal(out, eager):
            bad += 1
    assert bad == 0, f"{backend}: {bad} of 200 graph replays differ from eager (the FlashInfer #4841 / DeepGEMM #444 failure class)"
