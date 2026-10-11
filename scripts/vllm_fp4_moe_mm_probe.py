#!/usr/bin/env python3
"""vLLM's own grouped NVFP4 GEMM (`cutlass_fp4_moe_mm`, the MoE path on SM120) against the exact grouped oracle, with the
expert segments vLLM's routing produces (arbitrary lengths, so most segments start off a 128-row boundary): the natural control for
the FlashInfer cuDNN finding (`docs/flashinfer-grouped-fp4-rtx5090.md`).

The data preparation follows vLLM's `run_cutlass_moe_fp4` for GEMM 1: `get_cutlass_moe_mm_data` turns top-k ids into expert offsets,
problem sizes and padded block-scale offsets; `shuffle_rows` orders the tokens by expert; `scaled_fp4_experts_quant` quantises the
ordered tokens per expert with per-expert global scales and writes the scales swizzled per expert at the padded offsets. The oracle
reads the quantised codes and the scales back (un-swizzled per expert from its padded block) and computes the exact product of
each expert's rows against its weights; the kernel's output is then judged per element.

    PYTHONPATH=/path/to/sm120-fp4 python scripts/vllm_fp4_moe_mm_probe.py --out reports/vllm-fp4-moe-mm-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from sm120fp4 import oracle
from sm120fp4.layouts import from_128x4, to_128x4
from sm120fp4.reference import global_scale_for, quantize_nvfp4

CASES = (  # (tokens, experts, topk, n, k)
    (64, 4, 1, 512, 1024),
    (128, 8, 2, 1024, 2048),
    (32, 2, 1, 256, 512),
    (300, 8, 2, 512, 1024),
)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("reports") / f"vllm-fp4-moe-mm-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.json")
    a = ap.parse_args(argv)
    import vllm
    import vllm._custom_ops as ops
    dev = torch.device("cuda")
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0), "torch": torch.__version__,
              "vllm": vllm.__version__, "cases": []}
    gen = torch.Generator().manual_seed(20261010)
    for m, e_count, topk, n, k in CASES:
        rec = {"tokens": m, "experts": e_count, "topk": topk, "n": n, "k": k}
        try:
            x = torch.randn(m, k, generator=gen).to(torch.bfloat16).to(dev)
            w = torch.randn(e_count, n, k, generator=gen).to(torch.bfloat16).float()
            # routing: random top-k experts per token (distinct within a token)
            topk_ids = torch.stack([torch.randperm(e_count, generator=gen)[:topk] for _ in range(m)]).to(torch.int32).to(dev)
            expert_offsets = torch.empty(e_count + 1, dtype=torch.int32, device=dev)
            blockscale_offsets = torch.empty(e_count + 1, dtype=torch.int32, device=dev)
            problem_sizes1 = torch.empty((e_count, 3), dtype=torch.int32, device=dev)
            problem_sizes2 = torch.empty((e_count, 3), dtype=torch.int32, device=dev)
            a_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=dev)
            c_map = torch.empty(topk_ids.numel(), dtype=torch.int32, device=dev)
            ops.get_cutlass_moe_mm_data(topk_ids, expert_offsets, problem_sizes1, problem_sizes2, a_map, c_map, e_count, n, k, blockscale_offsets, is_gated=False)
            a_rows = ops.shuffle_rows(x, a_map)                                   # [m*topk, k] ordered by expert
            a1_gscale = torch.stack([global_scale_for(a_rows[expert_offsets[e]:expert_offsets[e + 1]].float()) if expert_offsets[e + 1] > expert_offsets[e] else torch.ones(1) for e in range(e_count)]).reshape(-1).to(dev)
            rep_a_fp4, rep_a_sf = ops.scaled_fp4_experts_quant(a_rows, a1_gscale, expert_offsets, blockscale_offsets, topk)
            # weights: reference quantiser per expert, one global scale per expert, swizzled per expert
            wq = [quantize_nvfp4(w[e]) for e in range(e_count)]
            w_fp4 = torch.stack([q[0] for q in wq]).to(dev)
            w_sf_lin = torch.stack([q[1] for q in wq])
            w_sf = torch.stack([to_128x4(q[1]) for q in wq]).view(torch.float8_e4m3fn).to(dev)
            w_gs = torch.stack([q[2] for q in wq]).reshape(-1)
            alphas = (1.0 / (a1_gscale.double().cpu() * w_gs.double())).float().to(dev)
            out = torch.empty(m * topk, n, dtype=torch.bfloat16, device=dev)
            ops.cutlass_fp4_moe_mm(out, rep_a_fp4, w_fp4, rep_a_sf, w_sf, alphas, problem_sizes1, expert_offsets[:-1], blockscale_offsets[:-1])
            torch.cuda.synchronize()
            # oracle: per expert, codes rows [eo, eo+1) and scales from the padded swizzled block at blockscale_offsets[e]
            eo = expert_offsets.cpu().tolist(); bo = blockscale_offsets.cpu().tolist()
            sfc = k // 16
            flat_sf = rep_a_sf.reshape(-1).view(torch.uint8).cpu()
            exact = torch.zeros(m * topk, n, dtype=torch.float64); bound = torch.zeros_like(exact)
            segs = []
            for e in range(e_count):
                rows = eo[e + 1] - eo[e]
                segs.append(rows)
                if rows == 0:
                    continue
                padded_rows = bo[e + 1] - bo[e] if e + 1 < len(bo) else ((rows + 127) // 128) * 128
                seg_sf_sw = flat_sf[bo[e] * sfc: bo[e] * sfc + padded_rows * sfc]
                seg_sf = from_128x4(seg_sf_sw, rows, sfc).reshape(rows, sfc) if padded_rows else None
                ex, bd = oracle.exact_gemm_nvfp4(rep_a_fp4[eo[e]:eo[e + 1]].cpu(), seg_sf, a1_gscale[e:e + 1].cpu(), w_fp4[e].cpu(), w_sf_lin[e], w_gs[e:e + 1], out_dtype=torch.bfloat16)
                exact[eo[e]:eo[e + 1]] = ex; bound[eo[e]:eo[e + 1]] = bd
            v = oracle.conformance(out.float(), exact, bound)
            err = (out.double().cpu() - exact).abs() > bound
            v["segments"] = segs
            v["segment_starts"] = eo[:-1]
            v["outside_bound_per_segment"] = [int(err[eo[e]:eo[e + 1]].sum()) for e in range(e_count)]
            rec.update(v)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:600])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
