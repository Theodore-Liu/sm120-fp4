"""Stage 2 baseline: the NVFP4 MoE paths FlashInfer already ships for SM120, at decode batch sizes, against an exact reference.

One MoE layer of Qwen3-30B-A3B's shape by default (128 experts, top-8, hidden 2048, expert intermediate 768, SwiGLU).
For each token count M the script builds one set of FP4 weights with the reference quantizer in this repository, hands
the same bytes to every backend in the layout that backend expects, and reports

- the normwise relative error ||out - ref|| / ||ref|| against the reference that performs the backend's own arithmetic
  (W4A4: activations quantized to NVFP4 before each GEMM; W4A16: activations kept in bf16), in fp32 on dequantized
  weights; and against the same reference without activation quantization, which is what the model "means";
- the median latency of the call (CUDA events, after warm-up), eager and replayed from a CUDA graph, the graph replay
  both with a warm L2 and with L2 flushed before each replay; and the cold replay against the time it takes just to read
  the weight bytes of the experts the batch touches at the card's DRAM bandwidth. The cold number is the one that
  describes decode: inside a forward pass the other layers' weights evict this layer's from L2 (96 MB on the RTX 5090,
  which holds the experts a batch of 1 to 4 tokens touches, so warm timings of small batches measure L2, not DRAM).

Backends: FlashInfer `b12x_fused_moe` with quant_mode "nvfp4" (W4A4) and "w4a16", `cutlass_fused_moe` (W4A4), and,
when vLLM is importable, vLLM's `fused_marlin_moe` (W4A16, the path vLLM uses for NVFP4 MoE on GPUs without a native FP4
MoE kernel), prepared with vLLM's own `prepare_nvfp4_moe_layer_for_marlin` from the same FP4 bytes. vLLM orders the
first weight matrix [gate ; up] and FlashInfer [up ; gate]; the Marlin copy has its halves swapped, codes and scales.

Scale convention: FlashInfer 0.6.16.post3's b12x path uses its first alpha both as the FC1 weight scale and as the input
quantization scale (later FlashInfer adds `input_global_scale` to separate them). The script therefore uses global scale
1 for weights, inputs and the FC2 input, with unit alphas, which is the convention of FlashInfer's own MoE tests.

    PYTHONPATH=. python scripts/bench_moe_baseline.py --out reports/moe-baseline-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from sm120fp4 import dequantize_nvfp4, quantize_nvfp4, to_128x4

ONE = torch.ones(1)


def build(e, h, i, device):
    g = torch.Generator().manual_seed(11)
    w1 = (torch.randn(e, 2 * i, h, generator=g) / 10).to(torch.bfloat16)   # rows: [up ; gate]
    w2 = (torch.randn(e, h, i, generator=g) / 10).to(torch.bfloat16)
    q1, s1, _ = quantize_nvfp4(w1.reshape(e * 2 * i, h).to(device), ONE.to(device))
    q2, s2, _ = quantize_nvfp4(w2.reshape(e * h, i).to(device), ONE.to(device))
    w1d = dequantize_nvfp4(q1, s1, ONE.to(device)).view(e, 2 * i, h)
    w2d = dequantize_nvfp4(q2, s2, ONE.to(device)).view(e, h, i)
    return dict(q1=q1.view(e, 2 * i, h // 2), s1=s1, q2=q2.view(e, h, i // 2), s2=s2, w1d=w1d, w2d=w2d)


def reference(x, w, ids, wts, i, act_quant, bf16_mid=False):
    """fp32 MoE on dequantized weights; act_quant=True quantizes each GEMM input to NVFP4 (global scale 1);
    bf16_mid=True rounds the FC1 output and the activation to bf16 before they are used, as a kernel that stores them in
    bf16 does."""
    dev = x.device
    r = (lambda t: t.to(torch.bfloat16).float()) if bf16_mid else (lambda t: t)

    def q(t):
        if not act_quant:
            return t.float()
        p, s, _ = quantize_nvfp4(t.float(), ONE.to(dev))
        return dequantize_nvfp4(p, s, ONE.to(dev))

    xq = q(x)
    out = torch.zeros(x.shape[0], w["w2d"].shape[1], device=dev)
    for t in range(x.shape[0]):
        for j in range(ids.shape[1]):
            ex = int(ids[t, j])
            hmid = r(xq[t:t + 1] @ w["w1d"][ex].T)
            a = r(F.silu(hmid[:, i:]) * hmid[:, :i])
            out[t] += float(wts[t, j]) * (q(a) @ w["w2d"][ex].T)[0]
    return out


_FLUSH = None


def flush_l2():
    """Overwrite a buffer larger than any SM120 part's L2 (RTX 5090: 96 MB), so the next call reads its weights from DRAM
    as the layer would inside a forward pass, where every other layer's weights pass through L2 in between."""
    global _FLUSH
    if _FLUSH is None:
        _FLUSH = torch.empty(256 * 2**20, dtype=torch.uint8, device="cuda")
    _FLUSH.fill_(1)


def timed(fn, warmup, iters, cold=False):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if cold:
            flush_l2()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    ts.sort()
    return ts[len(ts) // 2] * 1000.0  # us


def marlin_setup(w, e, h, i, dev):
    """vLLM Marlin W4A16 MoE from the same FP4 codes and E4M3 block scales, or (None, reason) without vLLM."""
    try:
        from types import SimpleNamespace

        from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
        from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
        from vllm.scalar_type import scalar_types
    except Exception as ex:  # noqa: BLE001
        return None, f"vLLM unavailable: {type(ex).__name__}: {str(ex)[:120]}"

    def gate_first(t):  # [E, 2i, ...] as [up ; gate] -> [gate ; up]
        return torch.cat([t[:, i:], t[:, :i]], dim=1).contiguous()

    layer = SimpleNamespace(num_experts=e, hidden_size=h, intermediate_size_per_partition=i, params_dtype=torch.bfloat16)
    w13 = gate_first(w["q1"])
    w13_s = gate_first(w["s1"].view(e, 2 * i, h // 16)).view(torch.float8_e4m3fn)
    w2_s = w["s2"].view(e, h, i // 16).view(torch.float8_e4m3fn)
    ones = torch.ones(e, device=dev)
    mw13, ms13, mg13, mw2, ms2, mg2 = prepare_nvfp4_moe_layer_for_marlin(layer, w13, w13_s, ones, w["q2"].contiguous(), w2_s,
                                                                         ones, is_act_and_mul=True)
    qid = scalar_types.float4_e2m1f.id

    def call(x, ids, wts, out):
        return fused_marlin_moe(x, mw13, mw2, None, None, ms13, ms2, wts, ids, qid, global_num_experts=e,
                                global_scale1=mg13, global_scale2=mg2, workspace=layer.workspace, output=out)
    return call, None


def graph_timed(fn, out_buf, warmup, iters):
    """Capture one call into a CUDA graph and time replays: the GPU's time without Python or launch overhead."""
    side = torch.cuda.Stream()
    with torch.cuda.stream(side):
        for _ in range(3):
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    eager = out_buf.clone()
    warm = timed(graph.replay, warmup, iters)
    cold = timed(graph.replay, warmup, iters, cold=True)
    graph.replay()
    torch.cuda.synchronize()
    return warm, cold, bool(torch.equal(out_buf, eager))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"moe-baseline-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    ap.add_argument("--experts", type=int, default=128)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=768)
    ap.add_argument("--tokens", default="1,2,4,8,16")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--dram-gbps", type=float, default=1792.0, help="card DRAM bandwidth in GB/s (RTX 5090: 1792)")
    a = ap.parse_args(argv)

    import flashinfer
    from flashinfer import fused_moe as fm
    from flashinfer.cute_dsl.utils import convert_sf_to_mma_layout

    dev = torch.device("cuda")
    e, k, h, i = a.experts, a.topk, a.hidden, a.inter
    w = build(e, h, i, dev)

    # b12x: 128x4-swizzled scales of the flattened [E*rows, K/16] matrix, then FlashInfer's 6-D MMA view.
    b12x_s1 = convert_sf_to_mma_layout(to_128x4(w["s1"]), m=2 * i, k=h, num_groups=e, sf_vec_size=16)
    b12x_s2 = convert_sf_to_mma_layout(to_128x4(w["s2"]), m=h, k=i, num_groups=e, sf_vec_size=16)
    # cutlass: per-expert 128x4-swizzled scales, [E, rows_pad128, cols_pad4], as FP8 bytes viewed int32.
    cut_s1 = torch.stack([to_128x4(w["s1"].view(e, 2 * i, h // 16)[x]) for x in range(e)]).view(torch.int32)
    cut_s2 = torch.stack([to_128x4(w["s2"].view(e, h, i // 16)[x]) for x in range(e)]).view(torch.int32)
    ones_e = torch.ones(e, device=dev)
    one = torch.ones(1, device=dev)
    scalar = torch.tensor(1.0, device=dev)  # cutlass_fused_moe wants the activation global scales 0-dimensional
    cut_scales = [scalar, cut_s1, ones_e, scalar, cut_s2, ones_e]

    marlin_call, marlin_why = marlin_setup(w, e, h, i, dev)
    if marlin_call is None:
        print(f"marlin-w4a16 skipped: {marlin_why}")

    # bytes of one expert's weights and scales: FC1 2i*h/2 + FC2 h*i/2 codes, and one E4M3 byte per 16 values
    expert_bytes = 3 * i * h // 2 + 3 * i * h // 16

    report = {"flashinfer": flashinfer.__version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
              "shape": {"experts": e, "top_k": k, "hidden": h, "inter": i}, "dram_gbps": a.dram_gbps,
              "expert_bytes": expert_bytes, "rows": []}
    for m in [int(t) for t in a.tokens.split(",")]:
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        logits = torch.randn(m, e, generator=g)
        wts, ids = torch.topk(F.softmax(logits, dim=-1), k, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev)
        ids = ids.to(torch.int32).to(dev)
        touched = int(torch.unique(ids).numel())
        floor_us = touched * expert_bytes / (a.dram_gbps * 1e3)
        ref_a4 = reference(x, w, ids, wts, i, act_quant=True)
        ref_a16 = reference(x, w, ids, wts, i, act_quant=False)
        ref_a4_bf16 = reference(x, w, ids, wts, i, act_quant=True, bf16_mid=True)
        ref_a16_bf16 = reference(x, w, ids, wts, i, act_quant=False, bf16_mid=True)
        outb = torch.empty(m, h, device=dev, dtype=torch.bfloat16)

        calls = {
            "b12x-nvfp4": lambda: fm.b12x_fused_moe(x, w["q1"], b12x_s1, w["q2"], b12x_s2, ids, wts, e, k, w1_alpha=ones_e,
                                                    w2_alpha=ones_e, fc2_input_scale=one, quant_mode="nvfp4", output=outb),
            "b12x-w4a16": lambda: fm.b12x_fused_moe(x, w["q1"], b12x_s1, w["q2"], b12x_s2, ids, wts, e, k, w1_alpha=ones_e,
                                                    w2_alpha=ones_e, fc2_input_scale=one, quant_mode="w4a16", output=outb),
            "cutlass-nvfp4": lambda: fm.cutlass_fused_moe(x, ids, wts, w["q1"].contiguous().view(torch.long),
                                                          w["q2"].contiguous().view(torch.long), torch.bfloat16,
                                                          quant_scales=cut_scales, output=outb),
        }
        if marlin_call is not None:
            calls["marlin-w4a16"] = lambda: marlin_call(x, ids, wts, outb)
        for name, fn in calls.items():
            row = {"backend": name, "tokens": m, "experts_touched": touched, "weight_read_floor_us": floor_us}
            try:
                out = fn()
                torch.cuda.synchronize()
                if isinstance(out, (list, tuple)):
                    out = out[0]
                o = out.float()
                ref = ref_a16 if name.endswith("w4a16") else ref_a4
                refb = ref_a16_bf16 if name.endswith("w4a16") else ref_a4_bf16
                row["rel_err_own_math"] = float((o - ref).norm() / ref.norm())
                row["rel_err_own_math_bf16_intermediates"] = float((o - refb).norm() / refb.norm())
                row["rel_err_vs_unquantized_activations"] = float((o - ref_a16).norm() / ref_a16.norm())
                row["all_finite"] = bool(torch.isfinite(o).all())
                row["median_us"] = timed(fn, a.warmup, a.iters)
                row["floor_fraction"] = floor_us / row["median_us"]
                try:
                    row["graph_median_us"], row["graph_cold_median_us"], row["graph_replay_equals_eager"] = graph_timed(fn, outb, a.warmup, a.iters)
                    row["graph_cold_floor_fraction"] = floor_us / row["graph_cold_median_us"]
                except Exception as gex:  # noqa: BLE001
                    row["graph_error"] = f"{type(gex).__name__}: {str(gex)[:200]}"
                row["ran"] = True
            except Exception as ex:  # noqa: BLE001
                row.update({"ran": False, "error": f"{type(ex).__name__}: {str(ex)[:300]}"})
            report["rows"].append(row)
            if row["ran"]:
                gm, gc = row.get("graph_median_us"), row.get("graph_cold_median_us")
                print(f"M={m:2d} {name:14s} eager {row['median_us']:7.1f} us  graph warm {gm if gm is None else round(gm, 1)} cold {gc if gc is None else round(gc, 1)} us  "
                      f"floor {floor_us:6.1f} us  err own {row['rel_err_own_math']:.4f} (bf16-mid {row['rel_err_own_math_bf16_intermediates']:.4f})  "
                      f"vs a16 {row['rel_err_vs_unquantized_activations']:.4f}  experts {touched}  {row.get('graph_error', '')[:80]}")
            else:
                print(f"M={m:2d} {name:14s} FAILED {row['error'][:160]}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
