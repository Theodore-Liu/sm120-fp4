"""The vLLM backend's weight path and forward against the layer the benches run (backlog item 2, step 4's unit test).

Three checks, all on layer 0 of nvidia/Qwen3-30B-A3B-NVFP4 read the way scripts/real_ckpt_layer.py reads it:

1. weights_from_vllm_layout, given the tensors in vLLM's layout ([gate; up]), produces byte-identical codes and scales to
   the loader's own [up; gate] stacking, and the same global scales.
2. The backend's forward is bit-identical to calling the kernels directly (what real_ckpt_layer.py times), at 1, 2, 4,
   8 and 16 tokens on random routing and on 8 fixed experts, and its error against the fp32 dequantized reference is
   within the real-weights table's bar.
3. A 40-token batch equals the 16+16+8 slices run one by one (the slicing fallback is the same function applied thrice).

Skipped without a CUDA device or without the checkpoint shard in the local HF cache. vLLM itself is not required: the
layout and forward code sit outside the vLLM classes, and the class definitions are exercised only when vLLM imports.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sm120fp4 import vllm_backend as vb  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")

E, H, I, K = 128, 2048, 768, 8
BAR = 0.01  # the real-weights table's rows read 0.29 to 0.35 percent against the fp32 reference


def _shard():
    try:
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        path = hf_hub_download("nvidia/Qwen3-30B-A3B-NVFP4", "model-00001-of-00004.safetensors", local_files_only=True)
    except Exception as exc:  # noqa: BLE001 - any failure means the shard is not local
        pytest.skip(f"checkpoint shard not in the local HF cache: {type(exc).__name__}")
    return safe_open(path, framework="pt", device="cpu")


@pytest.fixture(scope="module")
def layer0():
    fq = _shard()
    dev = torch.device("cuda")
    pre = "model.layers.0.mlp.experts."

    def fp4(ex, proj):
        b = f"{pre}{ex}.{proj}_proj."
        return (fq.get_tensor(b + "weight").to(dev), fq.get_tensor(b + "weight_scale").to(dev),
                fq.get_tensor(b + "weight_scale_2").float().reshape(()))

    gate_q, gate_s, gate_g, up_q, up_s, up_g, down_q, down_s, down_g = ([] for _ in range(9))
    for ex in range(E):
        q, s, g = fp4(ex, "gate"); gate_q.append(q); gate_s.append(s); gate_g.append(g)
        q, s, g = fp4(ex, "up"); up_q.append(q); up_s.append(s); up_g.append(g)
        q, s, g = fp4(ex, "down"); down_q.append(q); down_s.append(s); down_g.append(g)
    st = torch.stack
    # vLLM's layout: w13 = [gate; up]
    vllm = dict(w13=st([torch.cat([g, u], 0) for g, u in zip(gate_q, up_q)]),
                w13_scale=st([torch.cat([g, u], 0) for g, u in zip(gate_s, up_s)]),
                w13_scale_2=torch.stack([st(gate_g), st(up_g)], 1).to(dev),
                w2=st(down_q), w2_scale=st(down_s), w2_scale_2=st(down_g).to(dev))
    # the loader's layout (real_ckpt_layer.py): q1 = [up; gate]
    ours = dict(q1=st([torch.cat([u, g], 0) for g, u in zip(gate_q, up_q)]).contiguous(),
                s1=torch.cat([torch.cat([u, g], 0) for g, u in zip(gate_s, up_s)]).view(torch.uint8).contiguous(),
                q2=st(down_q).contiguous(), s2=torch.cat(down_s).view(torch.uint8).contiguous(),
                alpha1=st(up_g).to(dev), alpha2=st(down_g).to(dev))
    return vllm, ours


def test_layout_rotation_matches_the_loader(layer0):
    vllm, ours = layer0
    w = vb.weights_from_vllm_layout(vllm["w13"], vllm["w13_scale"], vllm["w13_scale_2"], vllm["w2"], vllm["w2_scale"],
                                    vllm["w2_scale_2"], K)
    assert (w.E, w.I, w.H, w.k) == (E, I, H, K)
    assert torch.equal(w.q1, ours["q1"])
    assert torch.equal(w.s1.reshape(-1, H // 16), ours["s1"])
    assert torch.equal(w.q2, ours["q2"])
    assert torch.equal(w.s2.reshape(-1, I // 16), ours["s2"])
    assert torch.equal(w.alpha1, ours["alpha1"]) and torch.equal(w.alpha2, ours["alpha2"])
    # gate == up global scale on every expert is what lets FC1 apply one alpha; the test also covers the refusal
    bad = vllm["w13_scale_2"].clone()
    bad[3, 1] *= 2
    with pytest.raises(ValueError, match="differ on 1 of 128"):
        vb.weights_from_vllm_layout(vllm["w13"], vllm["w13_scale"], bad, vllm["w2"], vllm["w2_scale"], vllm["w2_scale_2"], K)


def _inputs(m, routing, dev):
    g = torch.Generator().manual_seed(1000 + m)
    x = torch.randn(m, H, generator=g).to(device=dev, dtype=torch.bfloat16)
    if routing == "random":
        wts, ids = torch.topk(torch.softmax(torch.randn(m, E, generator=g), dim=-1), K, dim=-1)
        wts = (wts / wts.sum(-1, keepdim=True)).float().to(dev).contiguous()
        ids = ids.to(torch.int32).to(dev).contiguous()
    else:
        ids = (torch.arange(K, dtype=torch.int32, device=dev) * (E // K)).repeat(m, 1).contiguous()
        wts = torch.full((m, K), 1.0 / K, device=dev)
    return x, ids, wts


def _direct(k, ours, x, ids, wts):
    """The kernels called the way scripts/real_ckpt_layer.py calls them (its layer() closure, one slice)."""
    m = x.shape[0]
    f1, f2 = k.layer.choice(m)
    vb.set_pdl(k, k.layer.use_pdl(m))
    dev = x.device
    P, umax = m * K, min(E, m * K)
    experts = torch.empty(umax, dtype=torch.int32, device=dev)
    offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
    pairs = torch.empty(P, dtype=torch.int32, device=dev)
    act = torch.empty(P, I, device=dev, dtype=torch.bfloat16)
    out = torch.empty(m, H, device=dev, dtype=torch.bfloat16)
    scratch = torch.zeros(4 * 16 * H, device=dev)
    counters = torch.zeros(H // 16, dtype=torch.int32, device=dev)
    wflat = wts.reshape(-1).contiguous()
    k.route.route(ids, E, experts, offsets, pairs)
    if f1 == "cuda_core":
        k.fc1_cc.fc1_w4a16(ours["q1"], ours["s1"], x, experts, offsets, pairs, ours["alpha1"], act, I, K)
    else:
        k.fc1_tc.fc1_mma(ours["q1"], ours["s1"], x, experts, offsets, pairs, ours["alpha1"], act, I, K)
    if f2 == "cuda_core":
        k.fc2_cc.fc2_w4a16(ours["q2"], ours["s2"], act, experts, offsets, pairs, wflat, ours["alpha2"], out, K)
    else:
        k.fc2_pf.fc2_pf(ours["q2"], ours["s2"], act, experts, offsets, pairs, wflat, ours["alpha2"], out, scratch, counters, K, 1)
    torch.cuda.synchronize()
    return out


def _reference(ours, x, ids, wts):
    from sm120fp4 import dequantize_nvfp4
    dev = x.device
    one = torch.ones(1, device=dev)
    xf = x.float()
    out = torch.zeros(x.shape[0], H, device=dev)
    for t in range(x.shape[0]):
        for j in range(K):
            e = int(ids[t, j])
            q1, s1 = ours["q1"][e], ours["s1"].view(E, 2 * I, H // 16)[e]
            w1 = torch.cat([dequantize_nvfp4(q1[:I], s1[:I], one), dequantize_nvfp4(q1[I:], s1[I:], one)], 0) * ours["alpha1"][e]
            w2 = dequantize_nvfp4(ours["q2"][e], ours["s2"].view(E, H, I // 16)[e], one) * ours["alpha2"][e]
            h = w1 @ xf[t]
            a = torch.nn.functional.silu(h[I:]) * h[:I]  # rows [up; gate]: gate is the second half
            out[t] += float(wts[t, j]) * (w2 @ a)
    return out


@pytest.mark.parametrize("routing", ["random", "fixed8"])
@pytest.mark.parametrize("m", [1, 2, 4, 8, 16])
def test_forward_is_bit_identical_to_the_direct_kernel_calls(layer0, routing, m):
    vllm, ours = layer0
    k = vb.kernels()
    w = vb.weights_from_vllm_layout(vllm["w13"], vllm["w13_scale"], vllm["w13_scale_2"], vllm["w2"], vllm["w2_scale"],
                                    vllm["w2_scale_2"], K)
    x, ids, wts = _inputs(m, routing, torch.device("cuda"))
    out = w.forward(x, ids, wts)
    torch.cuda.synchronize()
    direct = _direct(k, ours, x, ids, wts)
    assert torch.equal(out, direct), "backend forward differs from the kernels called directly"
    ref = _reference(ours, x, ids, wts)
    rel = float((out.float() - ref).norm() / ref.norm())
    assert rel < BAR, f"relative error {rel:.4%} against the fp32 dequantized reference"


def test_long_batch_equals_its_slices(layer0):
    vllm, ours = layer0
    w = vb.weights_from_vllm_layout(vllm["w13"], vllm["w13_scale"], vllm["w13_scale_2"], vllm["w2"], vllm["w2_scale"],
                                    vllm["w2_scale_2"], K)
    x, ids, wts = _inputs(40, "random", torch.device("cuda"))
    out = w.forward(x, ids, wts)
    torch.cuda.synchronize()
    parts = [w.forward(x[s:s + 16], ids[s:s + 16], wts[s:s + 16]) for s in (0, 16, 32)]
    torch.cuda.synchronize()
    assert torch.equal(out, torch.cat(parts, 0))
    with pytest.raises(TypeError):
        w.forward(x.float(), ids, wts)


def test_vllm_classes_if_vllm_is_installed(monkeypatch):
    if importlib.util.find_spec("vllm") is None:
        pytest.skip("vLLM not installed")
    c = vb.classes()
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config, ModelOptNvFp4FusedMoE
    assert issubclass(c.Config, ModelOptNvFp4Config) and issubclass(c.Method, ModelOptNvFp4FusedMoE)
    monkeypatch.delenv(vb.ENV, raising=False)
    assert vb.register() is False
    monkeypatch.setenv(vb.ENV, "1")
    assert vb.register() is True
    from vllm.model_executor.layers.quantization import get_quantization_config
    assert get_quantization_config(vb.METHOD_NAME) is c.Config
