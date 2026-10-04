"""The indexer adapter against the v5 kernel's own reference, through vLLM's wrapper when vLLM is importable (backlog item 5,
step 1's unit test).

1. Flat (prefill): sm120fp4.indexer.fp8_fp4_mqa_logits on random e4m3 q and k with random [ks, ke) spans equals the kernel
   module's reference on the dequantised operands, at absolute columns, with -inf outside each row's span; the weights are
   fp32 with q's per-token scale folded in, the engine's form, so the test also reports the error the bf16 conversion of the
   weights costs against an fp32-weights reference (the number docs/stage3-engine-wiring.md 3d records).
2. Paged (decode): fp8_fp4_paged_mqa_logits on vLLM's [num_blocks, 64, 1, 132] cache, a [B, next_n, H, 128] q, 1-D and
   2-D context lengths and a block table with spare columns equals the flat call on the same rows.
3. Through vLLM: after register(force=True), vllm.utils.deep_gemm.fp8_fp4_mqa_logits / fp8_fp4_paged_mqa_logits return the
   adapter's results (skipped when vLLM is not installed in the interpreter running the test).

Skipped without a CUDA device.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sm120fp4 import indexer  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
DEV = torch.device("cuda")
BAR = 1e-5          # the kernel's own selftest bar against its reference (bf16 weights both sides)
REPORT = ROOT / "reports" / "indexer-adapter-weights-precision-rtx5090-20261004.json"


def _case(M, N, H, seed, full_span=False):
    """Engine-form operands: q e4m3 with a UE8M0 per-token scale folded into fp32 weights, k e4m3 with an fp32 row scale."""
    v5 = indexer._load_v5()
    q, kv, w, ks, ke = v5.base.make_case(M, N, H, seed, DEV, full_span)
    q8, sfq_packed = v5.ref.per_token_cast_to_fp8(q.reshape(M * H, v5.HEAD_DIM), use_ue8m0=True, gran_k=v5.HEAD_DIM,
                                                   use_packed_ue8m0=True)
    sfq_f = v5.ref.unpack_ue8m0_from_int(sfq_packed)[:, :1]
    # fp32 weights with full fp32 mantissas (make_case's are bf16, and a bf16 value times a power of two stays bf16-exact,
    # which would make the bf16-conversion measurement below vacuous); the UE8M0 per-token scale is folded in as the engine does
    g = torch.Generator().manual_seed(seed + 101)
    w32 = (torch.rand(M, H, generator=g).to(DEV) + 0.05) * sfq_f.reshape(M, H)
    w32 = w32.float().contiguous()
    q8 = q8.reshape(M, H, v5.HEAD_DIM).contiguous()
    k8, k_scale = v5.quantize_k_fp8(kv)
    return q8, k8, k_scale, w32, ks, ke


def _reference_abs(q8, k8, k_scale, w, ks, ke, N):
    """The kernel module's reference, with the weights as given (fp32 or bf16), laid out at absolute columns."""
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


def _rel(a, b):
    valid = torch.isfinite(b)
    return float((a[valid] - b[valid]).abs().max() / b[valid].abs().max().clamp_min(1e-30))


@pytest.mark.parametrize("M,N,H,seed", [(16, 512, 8, 1), (33, 2048, 8, 2), (64, 4096, 16, 3), (7, 300, 32, 4)])
def test_flat_matches_reference(M, N, H, seed):
    q8, k8, k_scale, w32, ks, ke = _case(M, N, H, seed)
    out = indexer.fp8_fp4_mqa_logits((q8, None), (k8, k_scale), w32, ks, ke, clean_logits=False)
    assert out.shape == (M, N) and out.dtype == torch.float32
    w_ref = w32 if indexer.weights_mode() == "fp32" else w32.to(torch.bfloat16)   # the reference reads the weights the adapter passed
    ref = _reference_abs(q8, k8, k_scale, w_ref, ks, ke, N)
    assert torch.equal(torch.isfinite(out), torch.isfinite(ref)), "logits outside [ks, ke) must stay -inf"
    assert _rel(out, ref) < BAR


def test_weights_precision_recorded(monkeypatch):
    """What each weights operand costs against an fp32-weights reference, both modes of the adapter, the largest relative
    error over the cases, written to the report the wiring doc cites. The fp32 mode (the default) is held to the kernel bar;
    the bf16 mode is recorded (it is the operand the measured kernel takes)."""
    rows = []
    for mode in ("bf16", "fp32"):
        monkeypatch.setenv(indexer.WEIGHTS_ENV, mode)
        for (M, N, H, seed) in ((16, 512, 8, 11), (33, 2048, 8, 12), (64, 4096, 16, 13), (48, 8192, 8, 14), (7, 300, 32, 15)):
            q8, k8, k_scale, w32, ks, ke = _case(M, N, H, seed)
            out = indexer.fp8_fp4_mqa_logits((q8, None), (k8, k_scale), w32, ks, ke, clean_logits=False)
            ref32 = _reference_abs(q8, k8, k_scale, w32, ks, ke, N)
            ref16 = _reference_abs(q8, k8, k_scale, w32.to(torch.bfloat16), ks, ke, N)
            rows.append({"weights_mode": mode, "M": M, "N": N, "H": H, "seed": seed, "rel_max_err_vs_fp32_weights": _rel(out, ref32),
                         "rel_max_err_vs_bf16_weights": _rel(out, ref16), "fp32_vs_bf16_reference": _rel(ref16, ref32)})
    worst = {m: max(r["rel_max_err_vs_fp32_weights"] for r in rows if r["weights_mode"] == m) for m in ("bf16", "fp32")}
    REPORT.parent.mkdir(exist_ok=True)
    REPORT.write_text(json.dumps({"device": torch.cuda.get_device_name(0),
                                  "note": "engine-form operands (q's UE8M0 per-token scale folded into fp32 weights with full mantissas); weights_mode bf16 converts them for the measured kernel, fp32 passes them to the f variant; rel_max_err is max |diff| / max |ref| over the valid logits",
                                  "rows": rows, "worst_rel_max_err_vs_fp32_weights": worst}, indent=1), encoding="utf-8")
    assert worst["fp32"] < BAR
    assert worst["bf16"] < 1e-2  # recorded; the wiring doc states the number


@pytest.mark.parametrize("B,next_n,N,H,seed,two_d", [(4, 1, 1000, 8, 21, False), (3, 2, 2048, 8, 22, True), (8, 1, 4096, 16, 23, True)])
def test_paged_matches_flat(B, next_n, N, H, seed, two_d):
    v5 = indexer._load_v5()
    S = B * next_n
    q8, k8, k_scale, w32, _, _ = _case(S, N, H, seed, full_span=True)
    g = torch.Generator().manual_seed(seed)
    ctx_rows = torch.randint(1, N + 1, (S,), generator=g).to(torch.int32).to(DEV)
    kv_cache, block_table, max_pages = v5.make_paged_fp8(k8, k_scale, S, ctx_rows, seed, DEV)
    # the engine's shapes: one block-table row per request (every row of a request shares it), with spare columns
    bt = torch.cat([block_table[::next_n], torch.zeros(B, 3, dtype=torch.int32, device=DEV)], dim=1)
    if two_d:
        ctx_arg = ctx_rows.reshape(B, next_n)
    else:
        ctx_rows = ctx_rows.reshape(B, next_n)[:, :1].expand(B, next_n).reshape(S).contiguous()
        ctx_arg = ctx_rows.reshape(B, next_n)[:, 0].contiguous()
    max_model_len = N + 64
    meta = indexer.get_paged_mqa_logits_metadata(ctx_arg, 64, 170)
    out = indexer.fp8_fp4_paged_mqa_logits((q8.reshape(B, next_n, H, v5.HEAD_DIM), None), kv_cache, w32, ctx_arg, bt, meta,
                                           max_model_len, clean_logits=False)
    assert out.shape == (S, max_model_len)
    ks0 = torch.zeros(S, dtype=torch.int32, device=DEV)
    flat = indexer.fp8_fp4_mqa_logits((q8, None), (k8, k_scale), w32, ks0, ctx_rows, clean_logits=False)
    assert torch.equal(out[:, :N], flat), "paged rows must equal the flat call on positions [0, ctx)"
    assert not torch.isfinite(out[:, N:]).any()


def test_through_vllm_wrapper():
    pytest.importorskip("vllm")
    import vllm.utils.deep_gemm as dg
    assert indexer.register(force=True)
    q8, k8, k_scale, w32, ks, ke = _case(16, 1024, 8, 31)
    via = dg.fp8_fp4_mqa_logits((q8, None), (k8, k_scale), w32, ks, ke, clean_logits=False)
    direct = indexer.fp8_fp4_mqa_logits((q8, None), (k8, k_scale), w32, ks, ke, clean_logits=False)
    assert torch.equal(via, direct)
    v5 = indexer._load_v5()
    B, next_n, H, N = 2, 2, 8, 512
    S = B * next_n
    q8, k8, k_scale, w32, _, _ = _case(S, N, H, 32, full_span=True)
    ctx = torch.randint(1, N + 1, (S,), generator=torch.Generator().manual_seed(32)).to(torch.int32).to(DEV)
    kv_cache, block_table, max_pages = v5.make_paged_fp8(k8, k_scale, S, ctx, 32, DEV)
    meta = dg.get_paged_mqa_logits_metadata(ctx.reshape(B, next_n), 64, 170)
    via = dg.fp8_fp4_paged_mqa_logits((q8.reshape(B, next_n, H, 128), None), kv_cache, w32, ctx.reshape(B, next_n),
                                      block_table[::next_n], meta, max_model_len=N, clean_logits=False)
    direct = indexer.fp8_fp4_paged_mqa_logits((q8.reshape(B, next_n, H, 128), None), kv_cache, w32, ctx.reshape(B, next_n),
                                              block_table[::next_n], meta, N, clean_logits=False)
    assert torch.equal(via, direct)
