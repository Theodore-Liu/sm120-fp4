"""The W4A16 decode MoE layer as an optional backend inside vLLM 0.28 (backlog item 2).

How it plugs in (docs/engine-integration-notes.md, section 5):

- vLLM resolves a ModelOpt NVFP4 checkpoint to the quantization method name ``modelopt_fp4`` through a fixed list of
  override names in ``vllm/config/model.py``, so a new name cannot claim such a checkpoint, and passing a different
  ``--quantization`` raises a mismatch error. ``register_quantization_config`` does let an installed package re-register
  an existing name, and ``get_quantization_config`` consults the customised registry first. So ``register()`` re-registers
  ``modelopt_fp4`` with :class:`SM120Fp4Config`, a subclass of vLLM's own config, when ``SM120FP4_MOE=1``; with the
  variable unset the registration is a no-op and vLLM is unchanged.
- :class:`SM120Fp4Config` hands every layer to vLLM's own methods except a routed-experts layer that
  :func:`SM120Fp4MoEMethod.fits`, which gets :class:`SM120Fp4MoEMethod`. A layer the kernels cannot take (a hidden size
  other than 2048, an intermediate size other than 768 or 1024, expert parallelism, a non-SwiGLU activation, shared
  experts) stays on vLLM's path, so one model can mix both.
- :class:`SM120Fp4MoEMethod` reuses the parent's ``create_weights`` (the checkpoint layout: FP4 codes ``[E, 2I, H/2]``
  and ``[E, H, I/2]``, e4m3 block scales, one ``weight_scale_2`` per expert and projection). ``process_weights_after_loading``
  does no Marlin repack: it rotates ``w13`` from vLLM's ``[gate; up]`` to the kernels' ``[up; gate]`` once, asserts the two
  FC1 global scales equal per expert (FC1 applies one alpha), and keeps the codes and scales as the kernels read them.
  ``apply`` runs the routing, FC1 and FC2 kernels of ``scripts/moe_layer.py`` under its ``choice``/``use_pdl`` rules, in
  slices of at most 16 tokens (the FC2 kernels' ``MAXM``); a longer batch is a loop over slices on the same resident
  weights (section 3 of the notes: correct by construction, slow for prefill, no second copy of the weights).

The kernels are the repository's ``scripts/*.py`` modules compiled with ``torch.utils.cpp_extension.load_inline`` (cached by
torch after the first build), so this module needs the repository checkout beside the package, as the benches do.

Nothing in vLLM is modified. The entry point ``vllm.general_plugins`` in ``pyproject.toml`` calls :func:`register` at
engine start in every process.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ENV = "SM120FP4_MOE"
METHOD_NAME = "modelopt_fp4"  # the name vLLM resolves ModelOpt NVFP4 checkpoints to; re-registered under the opt-in
MAXM = 16  # the FC2 kernels take at most 16 tokens per call
HIDDEN = 2048  # scripts/fc1_w4a16.py and fc1_mma.py are specialised to this hidden size
INTERMEDIATES = (768, 1024)  # scripts/fc2_mma_pf.py's shape check

_SCRIPTS = Path(__file__).resolve().parent / "kernels"  # the packaged kernels (since 2026-10-03); the checkout's scripts/ holds shims
_KERNELS: SimpleNamespace | None = None


def enabled() -> bool:
    return os.environ.get(ENV, "") == "1"


def kernels() -> SimpleNamespace:
    """Load sm120fp4/kernels/moe_layer.py (which loads and compiles the five kernel modules) once per process."""
    global _KERNELS
    if _KERNELS is None:
        if not (_SCRIPTS / "moe_layer.py").is_file():
            raise FileNotFoundError(f"the packaged kernels are missing: {_SCRIPTS} has no moe_layer.py")
        if str(_SCRIPTS) not in sys.path:
            sys.path.insert(0, str(_SCRIPTS))
        if "moe_layer" not in sys.modules:
            spec = importlib.util.spec_from_file_location("moe_layer", _SCRIPTS / "moe_layer.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules["moe_layer"] = mod
            spec.loader.exec_module(mod)
        lm = sys.modules["moe_layer"]
        _KERNELS = SimpleNamespace(layer=lm, route=lm.moe.build(), fc1_cc=lm.fc1.build(), fc2_cc=lm.fc2.build(),
                                   fc1_tc=lm.fc1m.build(), fc2_pf=lm.fc2p.build())
    return _KERNELS


def set_pdl(k: SimpleNamespace, on: bool) -> None:
    k.fc1_cc.fc1_set_pdl(on)
    k.fc2_cc.fc2_set_pdl(on)
    k.fc1_tc.fc1_mma_set_pdl(on)
    k.fc2_pf.fc2_pf_set_pdl(on)


def rotate_halves(t: torch.Tensor, inter: int) -> torch.Tensor:
    """[gate; up] -> [up; gate] along dim 1 (and back: the rotation is its own inverse at equal halves)."""
    return torch.cat([t[:, inter:], t[:, :inter]], dim=1).contiguous()


class Weights:
    """The layer's weights and work buffers as the kernels read them."""

    def __init__(self, q1: torch.Tensor, s1: torch.Tensor, alpha1: torch.Tensor, q2: torch.Tensor, s2: torch.Tensor,
                 alpha2: torch.Tensor, top_k: int):
        e_n, two_i, h2 = q1.shape
        self.E, self.I, self.H, self.k = e_n, two_i // 2, h2 * 2, top_k
        if self.H != HIDDEN or self.I not in INTERMEDIATES:
            raise ValueError(f"the kernels take hidden {HIDDEN} and intermediate {INTERMEDIATES}, not {self.H}/{self.I}")
        if q1.dtype != torch.uint8 or q2.dtype != torch.uint8 or s1.dtype != torch.uint8 or s2.dtype != torch.uint8:
            raise TypeError("codes and block scales must be uint8 views")
        if tuple(q2.shape) != (e_n, self.H, self.I // 2) or tuple(s1.shape) != (e_n, two_i, self.H // 16) \
                or tuple(s2.shape) != (e_n, self.H, self.I // 16):
            raise ValueError("checkpoint layout expected: q1 [E,2I,H/2] s1 [E,2I,H/16] q2 [E,H,I/2] s2 [E,H,I/16]")
        if alpha1.dtype != torch.float32 or alpha2.dtype != torch.float32 or alpha1.numel() != e_n or alpha2.numel() != e_n:
            raise TypeError("one float32 global scale per expert and projection")
        self.q1, self.s1, self.alpha1 = q1.contiguous(), s1.contiguous(), alpha1.contiguous()
        self.q2, self.s2, self.alpha2 = q2.contiguous(), s2.contiguous(), alpha2.contiguous()
        dev = q1.device
        umax, pmax = min(e_n, MAXM * top_k), MAXM * top_k
        self.experts = torch.empty(umax, dtype=torch.int32, device=dev)
        self.offsets = torch.empty(umax + 1, dtype=torch.int32, device=dev)
        self.pairs = torch.empty(pmax, dtype=torch.int32, device=dev)
        self.act = torch.empty(pmax, self.I, dtype=torch.bfloat16, device=dev)
        self.scratch = torch.zeros(4 * MAXM * self.H, dtype=torch.float32, device=dev)
        self.counters = torch.zeros(self.H // 16, dtype=torch.int32, device=dev)

    def forward(self, x: torch.Tensor, topk_ids: torch.Tensor, topk_weights: torch.Tensor, out: torch.Tensor | None = None,
                pdl: bool | None = None) -> torch.Tensor:
        """The MoE forward for a batch of any size: slices of at most 16 tokens through route -> FC1 -> FC2."""
        if x.dtype != torch.bfloat16 or x.dim() != 2 or x.shape[1] != self.H:
            raise TypeError(f"x must be bf16 [tokens, {self.H}]")
        m = x.shape[0]
        if tuple(topk_ids.shape) != (m, self.k) or tuple(topk_weights.shape) != (m, self.k):
            raise ValueError("topk_ids and topk_weights must be [tokens, top_k]")
        k = kernels()
        x = x.contiguous()
        ids = topk_ids.to(torch.int32).contiguous()
        wts = topk_weights.to(torch.float32).contiguous()
        if out is None:
            out = torch.empty_like(x)
        for s in range(0, m, MAXM):
            e = min(m, s + MAXM)
            self._slice(k, x[s:e], ids[s:e], wts[s:e], out[s:e], pdl)
        return out

    def _slice(self, k, x, ids, wts, out, pdl):
        m = x.shape[0]
        f1, f2 = k.layer.choice(m)
        set_pdl(k, k.layer.use_pdl(m) if pdl is None else pdl)
        P, umax = m * self.k, min(self.E, m * self.k)
        experts, offsets, pairs, act = self.experts[:umax], self.offsets[:umax + 1], self.pairs[:P], self.act[:P]
        wflat = wts.reshape(-1).contiguous()
        k.route.route(ids, self.E, experts, offsets, pairs)
        if f1 == "cuda_core":
            k.fc1_cc.fc1_w4a16(self.q1, self.s1, x, experts, offsets, pairs, self.alpha1, act, self.I, self.k)
        else:
            k.fc1_tc.fc1_mma(self.q1, self.s1, x, experts, offsets, pairs, self.alpha1, act, self.I, self.k)
        if f2 == "cuda_core":
            k.fc2_cc.fc2_w4a16(self.q2, self.s2, act, experts, offsets, pairs, wflat, self.alpha2, out, self.k)
        else:
            k.fc2_pf.fc2_pf(self.q2, self.s2, act, experts, offsets, pairs, wflat, self.alpha2, out, self.scratch,
                            self.counters, self.k, 1)


def weights_from_vllm_layout(w13: torch.Tensor, w13_scale: torch.Tensor, w13_scale_2: torch.Tensor, w2: torch.Tensor,
                             w2_scale: torch.Tensor, w2_scale_2: torch.Tensor, top_k: int) -> Weights:
    """Build :class:`Weights` from the tensors vLLM's loader fills: w13 [E, 2I, H/2] as [gate; up], its e4m3 block
    scales [E, 2I, H/16], w13_scale_2 [E, 2] (gate, up), w2 [E, H, I/2], w2 scales [E, H, I/16], w2_scale_2 [E]."""
    e_n, two_i = w13.shape[0], w13.shape[1]
    inter = two_i // 2
    g = w13_scale_2.reshape(e_n, -1).to(torch.float32)
    if g.shape[1] == 2 and not torch.equal(g[:, 0], g[:, 1]):
        n = int((g[:, 0] != g[:, 1]).sum())
        raise ValueError(f"gate and up global scales differ on {n} of {e_n} experts; FC1 applies one alpha per expert")
    alpha1 = g[:, 0].contiguous()
    return Weights(rotate_halves(w13, inter).view(torch.uint8),
                   rotate_halves(w13_scale, inter).view(torch.uint8),
                   alpha1, w2.view(torch.uint8), w2_scale.view(torch.uint8), w2_scale_2.to(torch.float32).reshape(-1), top_k)


# ----------------------------------------------------------------------------------------------------------------------
# vLLM classes live in sm120fp4.vllm_classes (module level, so vLLM can pickle the config into its engine-core process);
# they are imported lazily so this module and its layout/forward code load without vLLM.

def _vllm():
    from vllm.model_executor.layers.quantization import register_quantization_config
    return SimpleNamespace(register=register_quantization_config)


def classes() -> SimpleNamespace:
    """Return the vLLM config and method subclasses (defined in sm120fp4.vllm_classes)."""
    from sm120fp4 import vllm_classes as vc
    return SimpleNamespace(Config=vc.SM120Fp4Config, Method=vc.SM120Fp4MoEMethod)


def _log(msg: str) -> None:
    try:
        from vllm.logger import init_logger
        init_logger("vllm.sm120fp4").info("sm120fp4: %s", msg)  # under the vllm logger so the engine log shows it
    except Exception:  # noqa: BLE001 - logging must never fail the engine
        print("sm120fp4:", msg, file=sys.stderr)


def register() -> bool:
    """The ``vllm.general_plugins`` entry point. Re-registers ``modelopt_fp4`` with :class:`SM120Fp4Config` when
    ``SM120FP4_MOE=1``; returns whether it did."""
    if not enabled():
        return False
    v = _vllm()
    v.register(METHOD_NAME)(classes().Config)
    _log(f"{METHOD_NAME} now resolves to SM120Fp4Config ({ENV}=1); routed-experts layers the kernels fit run on them")
    return True
