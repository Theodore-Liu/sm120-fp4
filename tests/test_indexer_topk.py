"""The engine's own top-k kernels over the adapter's logits (backlog item 5, step 1's closing test).

vLLM's indexer does two things with the MQA logits: it computes them (the DeepGEMM entry points the adapter now serves) and it
selects the top-k positions per query row with its own CUDA kernels, `top_k_per_row_prefill` (flat, spans [ks, ke)) and
`top_k_per_row_decode` (paged, lengths per (request, next_n)). What the attention layer consumes is the index buffer those kernels
write, so the test that closes the wiring is: run the engine's top-k kernels on the adapter's logits and on the fp32 reference
logits, and compare the selected index sets row by row. The prefill kernel writes each row's indices relative to its `ks` (0-based
in the span); the decode kernel writes absolute positions in [0, context length). Ties are made negligible by random operands; a row whose span is shorter
than topk must select its whole span.

Skipped without a CUDA device or without vLLM's compiled ops in the interpreter.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sm120fp4 import indexer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = torch.device("cuda")


def _ops():
    pytest.importorskip("vllm")
    import vllm._custom_ops as ops
    if not hasattr(ops, "top_k_per_row_prefill") or not hasattr(ops, "top_k_per_row_decode"):
        pytest.skip("vLLM build without the indexer top-k kernels")
    return ops


def _case(M, N, H, seed, full_span=False):
    v5 = indexer._load_v5()
    q, kv, w, ks, ke = v5.base.make_case(M, N, H, seed, DEV, full_span)
    q8, sfq_packed = v5.ref.per_token_cast_to_fp8(q.reshape(M * H, v5.HEAD_DIM), use_ue8m0=True, gran_k=v5.HEAD_DIM,
                                                   use_packed_ue8m0=True)
    sfq_f = v5.ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
    g = torch.Generator().manual_seed(seed + 101)
    w32 = ((torch.rand(M, H, generator=g).to(DEV) + 0.05) * sfq_f.reshape(M, H)).float().contiguous()
    q8 = q8.reshape(M, H, v5.HEAD_DIM).contiguous()
    k8, k_scale = v5.quantize_k_fp8(kv)
    return q8, k8, k_scale, w32, ks, ke


def _reference_abs(q8, k8, k_scale, w, ks, ke, N):
    v5 = indexer._load_v5()
    M, H, _ = q8.shape
    ones = torch.ones(M * H, 1, device=DEV)
    max_k = int((ke - ks).max())
    compact = v5.reference(q8, ones, k8, k_scale, w, ks, ke, max_k)
    out = torch.full((M, N), float("-inf"), device=DEV)
    for i in range(M):
        a, b = int(ks[i]), int(ke[i])
        out[i, a:b] = compact[i, : b - a]
    return out


def _sets(idx: torch.Tensor):
    rows = []
    for r in idx.tolist():
        rows.append(sorted(x for x in r if x >= 0))
    return rows


@pytest.mark.parametrize("M,N,H,seed,topk", [(16, 4096, 8, 41, 2048), (33, 3000, 8, 42, 512), (8, 8192, 16, 43, 2048)])
def test_prefill_topk_matches_reference(M, N, H, seed, topk):
    ops = _ops()
    q8, k8, k_scale, w32, ks, ke = _case(M, N, H, seed)
    ours = indexer.fp8_fp4_mqa_logits((q8, None), (k8, k_scale), w32, ks, ke, clean_logits=False)
    ref = _reference_abs(q8, k8, k_scale, w32, ks, ke, N)
    picked = []
    for logits in (ours, ref):
        idx = torch.full((M, topk), -1, dtype=torch.int32, device=DEV)
        ops.top_k_per_row_prefill(logits, ks, ke, idx, M, logits.stride(0), logits.stride(1), topk)
        torch.cuda.synchronize()
        picked.append(_sets(idx))
    for i in range(M):
        span = int(ke[i] - ks[i])
        assert len(picked[0][i]) == min(span, topk), (i, span, len(picked[0][i]))
        assert all(0 <= x < span for x in picked[0][i]), "the prefill kernel writes indices relative to ks"
        assert picked[0][i] == picked[1][i], f"row {i}: the engine's top-k differs between our logits and the reference"


@pytest.mark.parametrize("B,next_n,N,H,seed,topk", [(4, 1, 4096, 8, 51, 2048), (3, 2, 3000, 8, 52, 512), (2, 2, 8192, 16, 53, 2048)])
def test_decode_topk_matches_reference(B, next_n, N, H, seed, topk):
    ops = _ops()
    v5 = indexer._load_v5()
    S = B * next_n
    q8, k8, k_scale, w32, _, _ = _case(S, N, H, seed, full_span=True)
    g = torch.Generator().manual_seed(seed)
    ctx = torch.randint(topk // 2, N + 1, (S,), generator=g).to(torch.int32).to(DEV)
    kv_cache, block_table, _ = v5.make_paged_fp8(k8, k_scale, S, ctx, seed, DEV)
    seq_lens = ctx.reshape(B, next_n)
    max_model_len = N + 64
    ours = indexer.fp8_fp4_paged_mqa_logits((q8.reshape(B, next_n, H, 128), None), kv_cache, w32, seq_lens,
                                            block_table[::next_n], indexer.get_paged_mqa_logits_metadata(seq_lens, 64, 170),
                                            max_model_len, clean_logits=False)
    ks0 = torch.zeros(S, dtype=torch.int32, device=DEV)
    ref_n = _reference_abs(q8, k8, k_scale, w32, ks0, ctx, N)
    ref = torch.full((S, max_model_len), float("-inf"), device=DEV)
    ref[:, :N] = ref_n
    picked = []
    for logits in (ours, ref):
        idx = torch.full((S, topk), -1, dtype=torch.int32, device=DEV)
        ops.top_k_per_row_decode(logits, next_n, seq_lens, idx, S, logits.stride(0), logits.stride(1), topk)
        torch.cuda.synchronize()
        picked.append(_sets(idx))
    for i in range(S):
        n = int(ctx[i])
        assert len(picked[0][i]) == min(n, topk), (i, n, len(picked[0][i]))
        assert all(0 <= x < n for x in picked[0][i])
        assert picked[0][i] == picked[1][i], f"row {i}: the engine's decode top-k differs between our logits and the reference"
