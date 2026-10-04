"""The sparse-attention indexer adapter: vLLM's three DeepGEMM entry points for the FP8 MQA-logits kernel, served by the v5
kernel on SM120 (backlog item 5, step 1).

vLLM 0.28 binds the indexer's kernels in ``vllm/utils/deep_gemm.py``: ``fp8_fp4_mqa_logits`` (prefill, flat k),
``fp8_fp4_paged_mqa_logits`` (decode, the paged fp8 cache) and ``get_paged_mqa_logits_metadata`` (the schedule the DeepGEMM
kernel wants; ours wants none). On SM120 vLLM refuses the MXFP4 indexer cache (``dsa_indexer_uses_fp4``), so the operands the
engine hands over are the FP8 ones: q [M, H, 128] e4m3 with its per-token scale folded into ``weights`` (fp32 [M, H]), k [N, 128]
e4m3 with one fp32 scale per row, and in decode the paged cache [num_blocks, 64, 1, 132] uint8 (128 e4m3 bytes then the fp32 scale
per row). The v5 kernel takes exactly these (its fp32-weights variant reads ``weights`` as the engine hands them over; the
bf16 form, selectable with ``SM120FP4_INDEXER_WEIGHTS=bf16``, costs up to 2.8e-3 relative on the logits) and writes the flat
logits compacted to [M, max(ke - ks)]; this module expands them to the engine's [M, N] with absolute columns
(-inf elsewhere), and reshapes the decode batch ([B, next_n, H, 128] q, 1-D or 2-D context lengths, one block-table row per
request) to the kernel's one-row-per-query form.

``register()`` binds the three ``_impl`` slots of ``vllm.utils.deep_gemm`` when ``SM120FP4_INDEXER=1``; the plugin's
``vllm_backend.register`` calls it. Nothing here imports vLLM at module load, so the functions are testable against the kernel's
own reference without the engine.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import threading
from pathlib import Path

import torch

ENV = "SM120FP4_INDEXER"
WEIGHTS_ENV = "SM120FP4_INDEXER_WEIGHTS"   # "fp32" (default: the engine's operand, read as is) or "bf16" (the measured kernel; up to 2.8e-3 relative)
_KERNELS = Path(__file__).resolve().parent / "kernels"
_V5_FILE = _KERNELS / "fp8_mqa_logits_v5_sm120.py"
PAGE = 64
ENTRY = 132
HEAD_DIM = 128

_lock = threading.Lock()
_v5 = None
_mod = None


def enabled() -> bool:
    return os.environ.get(ENV, "") == "1"


def weights_mode() -> str:
    m = os.environ.get(WEIGHTS_ENV, "fp32")
    if m not in ("bf16", "fp32"):
        raise ValueError(f"{WEIGHTS_ENV} must be bf16 or fp32, got {m!r}")
    return m


def _load_v5():
    """The v5 kernel module, loaded by path from the packaged kernels (it imports its siblings from the same directory)."""
    global _v5
    if _v5 is None:
        if not _V5_FILE.is_file():
            raise FileNotFoundError(f"the packaged kernels are missing: {_V5_FILE}")
        if str(_KERNELS) not in sys.path:
            sys.path.insert(0, str(_KERNELS))
        spec = importlib.util.spec_from_file_location("sm120fp4_fp8_mqa_logits_v5", _V5_FILE)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _v5 = mod
    return _v5


def build(verbose: bool = False):
    """Compile (once per process) and return the v5 extension; the first call pays the nvcc build."""
    global _mod
    with _lock:
        if _mod is None:
            _mod = _load_v5().build(verbose=verbose)
    return _mod


def _weights(weights: torch.Tensor, S: int, H: int) -> torch.Tensor:
    if weights.shape != (S, H):
        raise ValueError(f"weights: expected [{S}, {H}], got {tuple(weights.shape)}")
    return weights.to(torch.float32 if weights_mode() == "fp32" else torch.bfloat16).contiguous()


def _check_q(q, S_expected: int | None = None) -> torch.Tensor:
    q8, q_scale = q
    if q_scale is not None:
        raise NotImplementedError("sm120fp4 indexer: FP8 q only (vLLM refuses the MXFP4 indexer cache on SM120 itself)")
    if q8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"q: expected float8_e4m3fn, got {q8.dtype}")
    if q8.shape[-1] != HEAD_DIM:
        raise ValueError(f"q: head dim {q8.shape[-1]} (the kernel takes {HEAD_DIM})")
    return q8


# ----------------------------------------------------------------------------------------------------------------------
# prefill: flat k

def fp8_fp4_mqa_logits(q, kv, weights: torch.Tensor, cu_seqlen_ks: torch.Tensor, cu_seqlen_ke: torch.Tensor,
                       clean_logits: bool) -> torch.Tensor:
    """``vllm.utils.deep_gemm.fp8_fp4_mqa_logits`` for the FP8 path: q = (e4m3 [M, H, 128], None), kv = (e4m3 [N, 128],
    fp32 [N]); returns fp32 [M, N] with logits at absolute positions [ks[i], ke[i]) of row i and -inf elsewhere (whatever
    ``clean_logits`` says: the engine's top-k reads only the valid span)."""
    q8 = _check_q(q)
    k8, k_scale = kv
    if k8.dtype != torch.float8_e4m3fn:
        raise TypeError(f"k: expected float8_e4m3fn, got {k8.dtype}")
    M, H, _ = q8.shape
    N = k8.shape[0]
    dev = q8.device
    out = torch.full((M, N), float("-inf"), dtype=torch.float32, device=dev)
    if M == 0 or N == 0:
        return out
    ks = cu_seqlen_ks.to(torch.int32).contiguous()
    ke = cu_seqlen_ke.to(torch.int32).contiguous()
    lens = (ke - ks).clamp_min(0)
    max_k = int(lens.max())
    if max_k == 0:
        return out
    if int(ke.max()) > N or int(ks.min()) < 0:
        raise ValueError(f"cu_seqlen_ks/ke must lie in [0, {N}]")
    v5 = _load_v5()
    mod = build()
    sfq = torch.full((M, H), 127, dtype=torch.uint8, device=dev)   # UE8M0 one: the engine folds q's scale into weights
    w = _weights(weights, M, H)
    compact = torch.full((M, max_k), float("-inf"), dtype=torch.float32, device=dev)
    v5.launch_v5(mod, q8.contiguous(), sfq, k8.contiguous(), k_scale.to(torch.float32).contiguous(), w, ks, ke, compact)
    ar = torch.arange(max_k, device=dev, dtype=torch.int32)
    valid = ar.unsqueeze(0) < lens.unsqueeze(1)                       # [M, max_k]
    cols = (ks.unsqueeze(1) + ar.unsqueeze(0)).clamp_max(N - 1).to(torch.long)
    rows = torch.arange(M, device=dev).unsqueeze(1).expand(M, max_k)
    out[rows[valid], cols[valid]] = compact[valid]
    return out


# ----------------------------------------------------------------------------------------------------------------------
# decode: the paged fp8 cache

def get_paged_mqa_logits_metadata(context_lens: torch.Tensor, block_size: int, num_sms: int) -> torch.Tensor:
    """DeepGEMM's kernel wants a per-SM schedule; v5 schedules by (query row, page) in its grid and wants none. The engine
    stores and passes whatever this returns, so an empty int32 tensor is the whole contract."""
    return torch.empty(0, dtype=torch.int32, device=context_lens.device)


def fp8_fp4_paged_mqa_logits(q, kv_cache: torch.Tensor, weights: torch.Tensor, context_lens: torch.Tensor,
                             block_tables: torch.Tensor, schedule_metadata: torch.Tensor, max_model_len: int,
                             clean_logits: bool, indices: torch.Tensor | None = None) -> torch.Tensor:
    """``vllm.utils.deep_gemm.fp8_fp4_paged_mqa_logits`` for the FP8 path: q = (e4m3 [B, next_n, H, 128], None), the cache
    [num_blocks, block, 1, 132] uint8 with block a multiple of 64 (64 on the V3.2 indexer backend, 256 on V4's; a block is
    consecutive 64-row pages), weights fp32 [B * next_n, H], context_lens int32 [B] or [B, next_n], block_tables int32
    [B, max_blocks]; returns fp32 [B * next_n, max_model_len], row (b, j) holding positions [0, context_lens[b, j]) and -inf
    beyond. ``schedule_metadata`` is ignored (see ``get_paged_mqa_logits_metadata``); ``indices`` (vLLM's varlen row map)
    is not served: the engine takes that branch only with its own packing kernel, which the SM120 path does not use."""
    q8 = _check_q(q)
    if indices is not None:
        raise NotImplementedError("sm120fp4 indexer: the varlen (indices) form of the paged call is not served")
    if q8.dim() != 4:
        raise ValueError(f"q: expected [B, next_n, H, 128], got {tuple(q8.shape)}")
    B, next_n, H, _ = q8.shape
    S = B * next_n
    dev = q8.device
    if kv_cache.dtype != torch.uint8 or kv_cache.dim() != 4 or kv_cache.shape[1] % PAGE or kv_cache.shape[2] != 1 or kv_cache.shape[3] != ENTRY:
        raise ValueError(f"kv_cache: expected uint8 [num_blocks, k * {PAGE}, 1, {ENTRY}], got {kv_cache.dtype} {tuple(kv_cache.shape)}")
    sub = kv_cache.shape[1] // PAGE          # 64-row pages per engine block: 1 on the V3.2 indexer backend (block 64), 4 on V4 (block 256)
    if context_lens.dim() == 2:
        if tuple(context_lens.shape) != (B, next_n):
            raise ValueError(f"context_lens: expected [{B}, {next_n}], got {tuple(context_lens.shape)}")
        ctx = context_lens.reshape(S)
    else:
        if context_lens.numel() != B:
            raise ValueError(f"context_lens: expected [{B}], got {tuple(context_lens.shape)}")
        ctx = context_lens.repeat_interleave(next_n)
    ctx = ctx.to(torch.int32).contiguous()
    out = torch.full((S, max_model_len), float("-inf"), dtype=torch.float32, device=dev)
    if S == 0:
        return out
    max_ctx = int(ctx.max())
    if max_ctx == 0:
        return out
    if max_ctx > max_model_len:
        raise ValueError(f"context length {max_ctx} exceeds max_model_len {max_model_len}")
    max_pages = -(-max_ctx // PAGE)
    max_blocks = -(-max_pages // sub)
    if block_tables.shape[0] != B or block_tables.shape[1] < max_blocks:
        raise ValueError(f"block_tables: expected [{B}, >= {max_blocks}], got {tuple(block_tables.shape)}")
    bt = block_tables[:, :max_blocks].to(torch.int32)
    if sub > 1:
        # an engine block of sub * 64 rows is sub consecutive 64-row pages of the cache viewed as [num_blocks * sub, 64, 1, 132]
        bt = (bt.unsqueeze(2) * sub + torch.arange(sub, dtype=torch.int32, device=dev)).reshape(B, max_blocks * sub)[:, :max_pages]
        kv_cache = kv_cache.reshape(kv_cache.shape[0] * sub, PAGE, 1, ENTRY)
    bt = bt.repeat_interleave(next_n, dim=0).contiguous()   # [S, max_pages]
    v5 = _load_v5()
    mod = build()
    sfq = torch.full((S, H), 127, dtype=torch.uint8, device=dev)
    w = _weights(weights, S, H)
    v5.paged_fn(mod, w)(q8.reshape(S, H, HEAD_DIM).contiguous(), sfq, kv_cache.contiguous(), w, ctx, bt, out, max_pages)
    return out


# ----------------------------------------------------------------------------------------------------------------------
# binding into vLLM

def register(force: bool = False) -> bool:
    """Bind the three slots of ``vllm.utils.deep_gemm`` to this module when ``SM120FP4_INDEXER=1`` (or ``force``). Returns
    whether it did. The slots are the module-level ``_impl`` names its ``_lazy_init`` fills from DeepGEMM; filling them first
    makes ``_lazy_init`` a no-op for them (it returns once any of the three is set), so DeepGEMM's absence on SM120 no longer
    turns every indexer call into ``_missing``."""
    if not (force or enabled()):
        return False
    import vllm.utils.deep_gemm as dg
    dg._fp8_fp4_mqa_logits_impl = fp8_fp4_mqa_logits
    dg._fp8_fp4_paged_mqa_logits_impl = fp8_fp4_paged_mqa_logits
    dg._get_paged_mqa_logits_metadata_impl = get_paged_mqa_logits_metadata
    try:
        from vllm.logger import init_logger
        init_logger("vllm.sm120fp4").info("sm120fp4: the sparse-attention indexer's fp8 MQA logits now run on the v5 kernel (%s=1, weights %s)",
                                         ENV, weights_mode())
    except Exception:  # noqa: BLE001 - logging must never fail the engine
        print(f"sm120fp4: indexer bound to the v5 kernel ({ENV}=1)", file=sys.stderr)
    return True
