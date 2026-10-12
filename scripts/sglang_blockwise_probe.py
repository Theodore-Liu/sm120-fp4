#!/usr/bin/env python3
"""SGLang's JIT FP8 128-block-scaled GEMM for SM120 (`sglang.kernels.ops.gemm.fp8_blockwise_gemm.fp8_blockwise_scaled_mm`,
CUTLASS, `python/sglang/kernels/jit/csrc/gemm/fp8_blockwise/`) against the exact oracle.

Contract, from the host checks (`fp8_blockwise_scaled_mm_sm120.cuh`) and SGLang's own test
(`test/registered/kernels/ops/gemm/test_fp8_blockwise_gemm.py`): a e4m3 (m, k) row-major; b e4m3 (k, n) column-major (the
transposed view of a row-major (n, k)); k and n multiples of 16 bytes; out row a multiple of 16 bytes; scales fp32 with
scales_a.size(0) == m, scales_a.size(1) == k / 128 and scales_b == (k / 128, n / 128), integer division; the test builds the
scales column-major (`t().contiguous().t()`), A per token x 128, B per 128 x 128 block, and covers n, k in multiples of 128
only. The dispatcher picks a swapAB kernel with a 32-token tile for m <= 128 or m % 4 != 0, a 64-token swapAB tile for
128 < m <= 256, and a 128 x 128 kernel above.

Part 1, in contract (n, k multiples of 128): m in {1, 3, 4, 7, 64, 100, 128, 129, 200, 256, 257, 300, 1024} x (n, k) in
{(128, 512), (512, 1024), (4096, 4096), (1536, 7168)}, bf16, plus fp16 at (512, 1024); judged against the float64 product of
the expanded scales under the census bound; three runs each.

Part 2, shapes the checks accept but the 128-block scales do not cover (k or n off 128, scales sized by the checks' integer
division): the scale tensors are column-major views into larger buffers, run twice with different values in the buffers'
hidden tails. An output that changes with memory outside the tensors passed is undefined behaviour on an accepted call. The
output is also compared with two readings of the uncovered tail: the tail's products dropped, or the tail scaled like the last
covered block.

    PYTHONPATH=<sm120-fp4> python scripts/sglang_blockwise_probe.py --out reports/<file>.json   (SGLang installed)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

M_LIST = (1, 3, 4, 7, 64, 100, 128, 129, 200, 256, 257, 300, 1024)
NK_IN = ((128, 512), (512, 1024), (4096, 4096), (1536, 7168))
NK_OFF = ((128, 144), (136, 512), (136, 144), (1000, 1040))
M_OFF = (1, 7, 129, 300)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    from sglang.kernels.ops.gemm.fp8_blockwise_gemm import fp8_blockwise_scaled_mm
    from sm120fp4 import oracle
    dev = torch.device("cuda")
    e4m3 = torch.float8_e4m3fn
    report = {"generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "device": torch.cuda.get_device_name(0),
              "capability": list(torch.cuda.get_device_capability()), "torch": torch.__version__, "cases": []}

    def path_of(m):
        return "swapab-32" if (m <= 128 or m % 4) else ("swapab-64" if m <= 256 else "128x128")

    def bound(sabs, ex, k, dt):
        pre = k * oracle.U32 * sabs + 2 * oracle.U64 * sabs
        return pre + 0.5 * oracle._ulp(ex.abs() + pre, dt)

    def operands(m, n, k, g):
        aq = (torch.randn(m, k, generator=g) * 2).to(e4m3)
        bq = torch.randn(n, k, generator=g).to(e4m3)
        return aq, bq

    # ---------------------------------------------------------------- part 1
    cases = [(m, n, k, torch.bfloat16) for (n, k) in NK_IN for m in M_LIST] + [(m, 512, 1024, torch.float16) for m in M_LIST]
    for m, n, k, dt in cases:
        rec = {"part": "in-contract", "m": m, "n": n, "k": k, "out": str(dt).split(".")[-1], "path": path_of(m)}
        try:
            g = torch.Generator().manual_seed(20261012 + 41 * m + 3 * n + k)
            aq, bq = operands(m, n, k, g)
            sa = (torch.rand(m, k // 128, generator=g) * 0.02 + 0.005).float()
            sb = (torch.rand(k // 128, n // 128, generator=g) * 0.02 + 0.005).float()
            ad = aq.double() * sa.double().repeat_interleave(128, dim=1)
            bd = bq.double() * sb.double().t().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
            ex = ad @ bd.T
            bnd = bound(ad.abs() @ bd.abs().T, ex, k, dt)
            a_d = aq.to(dev)
            b_d = bq.to(dev).t()                                   # (k, n) column-major
            sa_d = sa.t().contiguous().t().to(dev)                 # column-major, as SGLang's test
            sb_d = sb.t().contiguous().t().to(dev)
            outs = []
            for _ in range(3):
                y = fp8_blockwise_scaled_mm(a_d, b_d, sa_d, sb_d, dt)
                torch.cuda.synchronize()
                outs.append(y.double().cpu())
            err = (outs[0] - ex).abs()
            rec.update({"elements": int(err.numel()), "outside": int((err > bnd).sum()), "nan": int(torch.isnan(outs[0]).sum()),
                        "max_err_over_bound": float((err / bnd.clamp(min=1e-300)).max()),
                        "distinct_outputs": len({t.numpy().tobytes() for t in outs})})
        except Exception as exc:  # noqa: BLE001
            rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        report["cases"].append(rec)
        print(json.dumps(rec)[:400], flush=True)

    # ---------------------------------------------------------------- part 2
    for n, k in NK_OFF:
        for m in M_OFF:
            rec = {"part": "accepted-off-128", "m": m, "n": n, "k": k, "path": path_of(m)}
            try:
                g = torch.Generator().manual_seed(20261013 + 41 * m + 3 * n + k)
                aq, bq = operands(m, n, k, g)
                kb, nb = k // 128, n // 128                       # what the checks require
                kb_full, nb_full = -(-k // 128), -(-n // 128)     # what the data spans
                sa_vals = (torch.rand(m, kb_full, generator=g) * 0.02 + 0.005).float()
                sb_vals = (torch.rand(kb_full, nb_full, generator=g) * 0.02 + 0.005).float()
                rec["scale_shapes_passed"] = [[m, kb], [kb, nb]]
                outs = {}
                for tag, fill in (("tail_A", 0.0), ("tail_B", 1000.0)):
                    # column-major (m, kb): storage (kb + 2, m); the view's columns are storage rows 0..kb-1
                    sto_a = torch.full((kb + 2, m), fill, dtype=torch.float32)
                    sto_a[:kb] = sa_vals[:, :kb].t()
                    # packed column-major (kb, nb): storage (nb + 2, kb), sentinel rows after the end only, so a kernel that
                    # assumes packed scales reads the passed values exactly and only reads past the end touch the sentinel
                    sto_b = torch.full((nb + 2, kb), fill, dtype=torch.float32)
                    sto_b[:nb] = sb_vals[:kb, :nb].t()
                    sa_d = sto_a.to(dev)[:kb].t()                  # (m, kb), strides (1, m)
                    sb_d = sto_b.to(dev)[:nb].t()                  # (kb, nb), strides (1, kb)
                    ys = []
                    for _ in range(2):
                        y = fp8_blockwise_scaled_mm(aq.to(dev), bq.to(dev).t(), sa_d, sb_d, torch.bfloat16)
                        torch.cuda.synchronize()
                        ys.append(y.double().cpu())
                    outs[tag] = ys
                rec["changes_with_hidden_memory"] = not torch.equal(outs["tail_A"][0], outs["tail_B"][0])
                rec["deterministic_given_memory"] = all(torch.equal(v[0], v[1]) for v in outs.values())
                y = outs["tail_A"][0]
                # readings of the uncovered tail
                sa_c = sa_vals[:, :kb].double().repeat_interleave(128, dim=1)
                sb_c = sb_vals[:kb, :nb].double().t().repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)
                kc, nc = kb * 128, nb * 128
                ex_drop = torch.zeros(m, n, dtype=torch.float64)
                ex_drop[:, :nc] = (aq[:, :kc].double() * sa_c) @ (bq[:nc, :kc].double() * sb_c).T
                sa_l = torch.cat([sa_c, sa_c[:, -1:].expand(m, k - kc)], 1) if k > kc else sa_c
                sb_l = sb_c
                if k > kc:
                    sb_l = torch.cat([sb_l, sb_l[:, -1:].expand(nc, k - kc)], 1)
                if n > nc:
                    sb_l = torch.cat([sb_l, sb_l[-1:, :].expand(n - nc, k)], 0)
                ex_last = (aq.double() * sa_l) @ (bq.double() * sb_l).T
                for name, ex in (("tail_dropped", ex_drop), ("tail_like_last_block", ex_last)):
                    bnd = bound((aq.double().abs() * sa_l.abs()) @ (bq.double().abs() * sb_l.abs()).T, ex, k, torch.bfloat16)
                    rec[f"outside_vs_{name}"] = int(((y - ex).abs() > bnd).sum())
                rec["elements"] = int(y.numel())
                rec["nan"] = int(torch.isnan(y).sum())
            except Exception as exc:  # noqa: BLE001
                rec["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
            report["cases"].append(rec)
            print(json.dumps(rec)[:500], flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    p1 = [c for c in report["cases"] if c["part"] == "in-contract" and "outside" in c]
    p2 = [c for c in report["cases"] if c["part"] == "accepted-off-128" and "elements" in c]
    print(f"in-contract judged {len(p1)}, outside in {sum(1 for c in p1 if c['outside'])}, nondet {sum(1 for c in p1 if c['distinct_outputs'] != 1)}; "
          f"off-128 accepted {len(p2)}, memory-dependent {sum(1 for c in p2 if c['changes_with_hidden_memory'])}; "
          f"errors {sum(1 for c in report['cases'] if 'error' in c)}; wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
