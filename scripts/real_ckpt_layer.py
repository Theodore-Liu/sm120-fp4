"""Stage 2 on real weights: layer 0's MoE experts of nvidia/Qwen3-30B-A3B-NVFP4 (ModelOpt NVFP4) through the layer.

1. Layout check against an independent source: each expert's gate, up and down projection is dequantized with this
   library's convention (E2M1 codes, even element in the low nibble; E4M3 block scales per 16, row-major; value =
   code x block scale x weight_scale_2) and compared with the same weights of the original bf16 checkpoint
   (Qwen/Qwen3-30B-A3B). The same comparison with the nibbles swapped must be far worse, so the check can fail.
2. The checkpoint's per-tensor global scales: FC1 applies one alpha per expert to both halves (up and gate), so the
   report records whether gate and up carry the same weight_scale_2.
3. The layer (router, FC1, FC2 chosen by batch size as scripts/moe_layer.py, PDL) on those weights at 1 to 16 randomly
   routed tokens: normwise error against an fp32 MoE on the dequantized weights, bit-identical over 50 calls, and
   graph-replay time with L2 flushed beside vLLM's Marlin W4A16 MoE on the same weights, in the same session.

    PYTHONPATH=. python scripts/real_ckpt_layer.py --out reports/real-ckpt-layer0-<device>-<date>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_here = Path(__file__).resolve().parent
for _name in ("bench_moe_baseline", "micro_floor", "fc1_w4a16", "fc2_w4a16", "fc1_mma", "fc2_mma", "fc2_mma_pf",
              "moe_w4a16", "moe_layer"):
    _spec = importlib.util.spec_from_file_location(_name, _here / f"{_name}.py")
    _mod = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _mod
    _spec.loader.exec_module(_mod)
bench = sys.modules["bench_moe_baseline"]
floor = sys.modules["micro_floor"]
fc1 = sys.modules["fc1_w4a16"]
fc2 = sys.modules["fc2_w4a16"]
fc1m = sys.modules["fc1_mma"]
fc2p = sys.modules["fc2_mma_pf"]
moe = sys.modules["moe_w4a16"]
layer_mod = sys.modules["moe_layer"]
from sm120fp4 import dequantize_nvfp4  # noqa: E402

FP4_REPO, FP4_FILE = "nvidia/Qwen3-30B-A3B-NVFP4", "model-00001-of-00004.safetensors"
BF16_REPO, BF16_FILE = "Qwen/Qwen3-30B-A3B", "model-00001-of-00016.safetensors"
LAYER = 0


def local(repo: str, fname: str) -> str:
    from huggingface_hub import hf_hub_download
    return hf_hub_download(repo, fname, local_files_only=True)


def swap_nibbles(packed: torch.Tensor) -> torch.Tensor:
    return ((packed & 0x0F) << 4) | (packed >> 4)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--with-cols32", action="store_true",
                    help="also time the layer with FC2 at 32 columns per block (scripts/fc2_cols32.py, 4 groups) from 4 "
                         "tokens up, beside the current choice and Marlin in the same session, on random and on "
                         "concentrated routing (16 tokens all to 8 experts)")
    a = ap.parse_args(argv)
    from safetensors import safe_open
    dev = torch.device("cuda")
    fq = safe_open(local(FP4_REPO, FP4_FILE), framework="pt", device="cpu")
    fb = safe_open(local(BF16_REPO, BF16_FILE), framework="pt", device="cpu")
    e_n, h, i, k = 128, 2048, 768, 8
    pre = f"model.layers.{LAYER}.mlp.experts."

    def fp4(ex, proj):
        b = f"{pre}{ex}.{proj}_proj."
        return (fq.get_tensor(b + "weight").to(dev), fq.get_tensor(b + "weight_scale").view(torch.uint8).to(dev),
                float(fq.get_tensor(b + "weight_scale_2").float()))

    # 1 and 2: layout against bf16, global scales
    errs = {"gate": [], "up": [], "down": []}
    errs_swapped = {"gate": [], "up": [], "down": []}
    gate_up_scale_equal = 0
    for ex in range(e_n):
        ws2 = {}
        for proj in ("gate", "up", "down"):
            q, s, g2 = fp4(ex, proj)
            ws2[proj] = g2
            ref = fb.get_tensor(f"{pre}{ex}.{proj}_proj.weight").float().to(dev)
            gs = torch.tensor(1.0 / g2, device=dev)
            d = dequantize_nvfp4(q, s, gs)
            errs[proj].append(float((d - ref).norm() / ref.norm()))
            if ex < 8:
                ds = dequantize_nvfp4(swap_nibbles(q), s, gs)
                errs_swapped[proj].append(float((ds - ref).norm() / ref.norm()))
        gate_up_scale_equal += int(ws2["gate"] == ws2["up"])
    layout = {p: {"median": sorted(v)[len(v) // 2], "max": max(v)} for p, v in errs.items()}
    layout_swapped = {p: {"median": sorted(v)[len(v) // 2]} for p, v in errs_swapped.items()}
    print("layout vs bf16:", json.dumps(layout), "\nswapped nibbles:", json.dumps(layout_swapped), flush=True)
    print("experts with gate and up weight_scale_2 equal:", gate_up_scale_equal, "of", e_n, flush=True)
    report = {"device": torch.cuda.get_device_name(0), "checkpoint": FP4_REPO, "file": FP4_FILE, "reference": BF16_REPO,
              "layer": LAYER, "layout_rel_err_vs_bf16": layout, "layout_rel_err_nibbles_swapped": layout_swapped,
              "gate_up_weight_scale_2_equal": gate_up_scale_equal, "experts": e_n, "rows": []}
    if gate_up_scale_equal != e_n:
        report["note"] = "gate and up global scales differ on some experts: FC1's single alpha cannot represent them"
        args_out(a.out, report)
        return 1

    # our layout: q1 [E, 2I, H/2] rows [up ; gate], s1 [E*2I, H/16]; q2 [E, H, I/2], s2 [E*H, I/16]; alphas per expert
    q1, s1, q2, s2, al1, al2, w1d, w2d = [], [], [], [], [], [], [], []
    for ex in range(e_n):
        qu, su, gu = fp4(ex, "up")
        qg, sg, gg = fp4(ex, "gate")
        qd, sd, gd = fp4(ex, "down")
        q1.append(torch.cat([qu, qg], 0))
        s1.append(torch.cat([su, sg], 0))
        q2.append(qd)
        s2.append(sd)
        al1.append(gu)
        al2.append(gd)
        one = torch.ones(1, device=dev)
        w1d.append(torch.cat([dequantize_nvfp4(qu, su, one), dequantize_nvfp4(qg, sg, one)], 0) * gu)
        w2d.append(dequantize_nvfp4(qd, sd, one) * gd)
    w = {"q1": torch.stack(q1).contiguous(), "s1": torch.cat(s1).contiguous(), "q2": torch.stack(q2).contiguous(),
         "s2": torch.cat(s2).contiguous(), "w1d": torch.stack(w1d), "w2d": torch.stack(w2d)}
    alpha1 = torch.tensor(al1, device=dev, dtype=torch.float32)
    alpha2 = torch.tensor(al2, device=dev, dtype=torch.float32)

    mr, m1, m2, m1m, m2p = moe.build(), fc1.build(), fc2.build(), fc1m.build(), fc2p.build()
    for fn in (m1.fc1_set_pdl, m2.fc2_set_pdl, m1m.fc1_mma_set_pdl, m2p.fc2_pf_set_pdl):
        fn(True)
    scratch = torch.zeros(4 * 16 * h, device=dev)
    counters = torch.zeros(h // 16, dtype=torch.int32, device=dev)

    # Marlin on the same codes and scales, with the checkpoint's own global scales
    from types import SimpleNamespace
    from vllm.model_executor.layers.fused_moe.experts.marlin_moe import fused_marlin_moe
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import prepare_nvfp4_moe_layer_for_marlin
    from vllm.scalar_type import scalar_types

    def gate_first(t):
        return torch.cat([t[:, i:], t[:, :i]], dim=1).contiguous()
    lay = SimpleNamespace(num_experts=e_n, hidden_size=h, intermediate_size_per_partition=i, params_dtype=torch.bfloat16)
    mw13, ms13, mg13, mw2, ms2, mg2 = prepare_nvfp4_moe_layer_for_marlin(
        lay, gate_first(w["q1"]), gate_first(w["s1"].view(e_n, 2 * i, h // 16)).view(torch.float8_e4m3fn), alpha1,
        w["q2"], w["s2"].view(e_n, h, i // 16).view(torch.float8_e4m3fn), alpha2, is_act_and_mul=True)
    qid = scalar_types.float4_e2m1f.id

    c32 = None
    if a.with_cols32:
        spec = importlib.util.spec_from_file_location("fc2_cols32", _here / "fc2_cols32.py")
        c32mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(c32mod)
        c32 = c32mod.build()
        c32.fc2_c32_set_pdl(True)
    cases = [("random", m) for m in (1, 2, 4, 8, 16)] + ([("fixed8", m) for m in (4, 16)] if a.with_cols32 else [])
    for routing, m in cases:
        g = torch.Generator().manual_seed(1000 + m)
        x = torch.randn(m, h, generator=g).to(device=dev, dtype=torch.bfloat16)
        if routing == "random":
            wts, ids = torch.topk(F.softmax(torch.randn(m, e_n, generator=g), dim=-1), k, dim=-1)
            wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
            ids = ids.to(torch.int32).to(dev).contiguous()
        else:
            ids = (torch.arange(k, dtype=torch.int32, device=dev) * (e_n // k)).repeat(m, 1).contiguous()
            wts = torch.full((m, k), 1.0 / k, device=dev)
        P, umax = m * k, min(e_n, m * k)
        experts = torch.empty(umax, dtype=torch.int32, device=dev)
        offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
        pairs = torch.empty(P, dtype=torch.int32, device=dev)
        act = torch.empty(P, i, device=dev, dtype=torch.bfloat16)
        out = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        outm = torch.empty(m, h, device=dev, dtype=torch.bfloat16)
        wflat = wts.reshape(-1).contiguous()
        f1, f2 = layer_mod.choice(m)

        def layer(fc2_cols32: bool = False):
            mr.route(ids, e_n, experts, offsets, pairs)
            if f1 == "cuda_core":
                m1.fc1_w4a16(w["q1"], w["s1"], x, experts, offsets, pairs, alpha1, act, i, k)
            else:
                m1m.fc1_mma(w["q1"], w["s1"], x, experts, offsets, pairs, alpha1, act, i, k)
            if fc2_cols32:
                c32.fc2_c32(w["q2"], w["s2"], act, experts, offsets, pairs, wflat, alpha2, out, scratch, counters, k, 4)
            elif f2 == "cuda_core":
                m2.fc2_w4a16(w["q2"], w["s2"], act, experts, offsets, pairs, wflat, alpha2, out, k)
            else:
                m2p.fc2_pf(w["q2"], w["s2"], act, experts, offsets, pairs, wflat, alpha2, out, scratch, counters, k,
                           1 if f2 == "prefetch" else 2)

        def marlin():
            fused_marlin_moe(x, mw13, mw2, None, None, ms13, ms2, wts, ids, qid, global_num_experts=e_n,
                             global_scale1=mg13, global_scale2=mg2, workspace=lay.workspace, output=outm)

        layer()
        marlin()
        torch.cuda.synchronize()
        ref = bench.reference(x, w, ids, wts, i, act_quant=False)
        rel = float((out.float() - ref).norm() / ref.norm())
        rel_m = float((outm.float() - ref).norm() / ref.norm())
        first = out.clone()
        stable = True
        for _ in range(50):
            layer()
            stable = stable and bool(torch.equal(out, first))
        row = {"tokens": m, "fc1": f1, "fc2": f2, "rel_err": rel, "marlin_rel_err": rel_m, "bit_identical_50": stable,
               "layer_us": floor.graph_time(layer), "marlin_us": floor.graph_time(marlin)}
        if a.with_cols32:
            row["routing"] = routing
            if m >= 4:
                layer(True)
                torch.cuda.synchronize()
                row["c32_rel_err"] = float((out.float() - ref).norm() / ref.norm())
                first32 = out.clone()
                ok32 = True
                for _ in range(50):
                    layer(True)
                    ok32 = ok32 and bool(torch.equal(out, first32))
                row["c32_bit_identical_50"] = ok32
                row["layer_c32_us"] = floor.graph_time(lambda: layer(True))
        report["rows"].append(row)
        print(json.dumps({kk: (round(v, 5) if isinstance(v, float) else v) for kk, v in row.items()}), flush=True)
    args_out(a.out, report)
    return 0


def args_out(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(f"written {path}")


if __name__ == "__main__":
    raise SystemExit(main())
