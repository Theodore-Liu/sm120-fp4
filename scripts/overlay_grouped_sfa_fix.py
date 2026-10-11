#!/usr/bin/env python3
"""Apply the grouped block-scale token-scale fix from a FlashInfer worktree to an installed FlashInfer package copy, so the
fix can be tested against a released wheel's compiled pieces without rebuilding FlashInfer main.

    python scripts/overlay_grouped_sfa_fix.py <worktree core.py> <overlay core.py>

The helper functions are copied verbatim from the worktree (between `def _segmented_token_scale_capacity_rows` and
`def _build_cudnn_moe_block_scale_grouped_gemm_graph`), and the two runner edits are applied by the same anchors.
"""
import ast
import sys

src_path, dst_path = sys.argv[1], sys.argv[2]
src = open(src_path, encoding="utf-8").read()
dst = open(dst_path, encoding="utf-8").read()
start = src.index("def _segmented_token_scale_capacity_rows(")
# The builder is decorated (@functools.lru_cache): the helpers go above the decorator, never between it and the def,
# or the cache wraps the wrong function and every call rebuilds the cuDNN graph (measured 2026-10-11: 25 ms a call).
anchor = "@functools.lru_cache(maxsize=1024)\ndef _build_cudnn_moe_block_scale_grouped_gemm_graph("
end = src.index(anchor)
helper = src[start:end]
assert dst.count(anchor) == 1, "decorated builder anchor"
assert "_segment_token_descale_per_group" not in dst
dst = dst.replace(anchor, helper + anchor)

for name, call in (
    ("_run_cudnn_moe_block_scale_grouped_gemm_mxfp8", "a.shape[1] // 32  # MXFP8: one UE8M0 scale per 32"),
    ("_run_cudnn_moe_block_scale_grouped_gemm_fp4", "a.shape[1] * 2 // block_size  # packed FP4: two per byte"),
):
    i = dst.index("def " + name + "(")
    nxt = dst.find("\ndef ", i + 10)
    j = len(dst) if nxt < 0 else nxt
    body = dst[i:j]
    old_decl = "    token_descale_3d = a_descale.unsqueeze(0)\n"
    assert body.count(old_decl) == 1, name
    body = body.replace(old_decl, old_decl + (
        "    token_descale_runtime = _bind_as_declared(\n"
        "        _segment_token_descale_per_group(\n"
        f"            a_descale, m_indptr, cum_m, {call}\n"
        "        ),\n"
        "        a_descale,\n"
        "    ).unsqueeze(0)\n"))
    old_vp = "        _CUDNN_UIDs.TOKEN_SCALE_FACTOR.value: token_descale_3d,\n"
    assert body.count(old_vp) == 1, name
    body = body.replace(old_vp, "        _CUDNN_UIDs.TOKEN_SCALE_FACTOR.value: token_descale_runtime,\n")
    dst = dst[:i] + body + dst[j:]

ast.parse(dst)
open(dst_path, "w", encoding="utf-8", newline="\n").write(dst)
print("overlay patched")

# The helper launches a Triton kernel that lives in flashinfer/triton/kernels/grouped_scale.py: copy it beside the runner too.
import os
import shutil

_root_src = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(src_path))))
_root_dst = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(dst_path))))
_k_src = os.path.join(_root_src, "triton", "kernels", "grouped_scale.py")
_k_dst = os.path.join(_root_dst, "triton", "kernels", "grouped_scale.py")
if os.path.exists(_k_src):
    shutil.copyfile(_k_src, _k_dst)
    print("kernel copied to", _k_dst)
