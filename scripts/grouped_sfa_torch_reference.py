import torch


def _segmented_token_scale_capacity_rows(total_rows: int, num_groups: int) -> int:
    """Rows of the per-group-padded token scale blob cuDNN's MoE block-scale kernels read.

    Mirrors cuDNN frontend's ``segmented_row_scale_capacity_rows``: with ``total_rows``
    tokens split over at most ``num_groups`` non-empty groups, independently padding
    every group to whole 128-row blocks needs at most ``A + (total_rows - A) // 128``
    blocks, ``A = min(total_rows, num_groups)``. The bound is static, so the blob can
    be sized without reading ``m_indptr`` on the host.
    """
    active = min(total_rows, num_groups)
    return 128 * (active + (total_rows - active) // 128)


def _segment_token_descale_per_group(
    a_descale: torch.Tensor, m_indptr: torch.Tensor, cum_m: int, k_blocks: int
) -> torch.Tensor:
    """Re-lay A's block scales from one 128x4 swizzle over all ``cum_m`` tokens into
    the layout cuDNN's MoE block-scale kernels read.

    ``grouped_mm_fp4`` / ``grouped_mm_mxfp8`` take ``a_descale`` as one F8_128x4-reordered
    tensor over every token, which is what the quantizers produce. cuDNN's MoE grouped
    block-scale kernels (SM100 and SM12x) instead read token scales that are
    F8_128x4-reordered and padded to whole 128-row blocks *per group*, groups
    concatenated in order: group ``g`` starts at block ``sum(ceil(rows_g' / 128))`` over
    the groups before it, and its rows are addressed group-locally. The two layouts
    coincide only when every group starts on a multiple of 128 rows; for any other
    ``m_indptr`` the kernel reads the wrong scales (or past the buffer) and returns
    wrong values without an error.

    The destination row of every source row follows from ``m_indptr`` on the device,
    so this needs no host synchronization and is safe under CUDA graph capture.
    Padding rows are zero. Returns a dense ``(capacity_rows, round_up(k_blocks, 4))``
    tensor of ``a_descale``'s dtype.
    """
    cols = ((k_blocks + 3) // 4) * 4
    rows = a_descale.numel() // cols
    num_groups = m_indptr.numel() - 1
    cap = _segmented_token_scale_capacity_rows(cum_m, num_groups)
    src_bytes = a_descale.contiguous().view(torch.uint8).reshape(rows, cols)
    linear = (
        src_bytes.reshape(rows // 128, cols // 4, 32, 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(rows, cols)
    )
    indptr = m_indptr.to(torch.int64)
    counts = indptr[1:] - indptr[:-1]
    padded = (counts + 127) // 128 * 128
    padded_end = torch.cumsum(padded, 0)
    padded_start = padded_end - padded
    dst = torch.arange(cap, dtype=torch.int64, device=a_descale.device)
    group = torch.searchsorted(padded_end, dst, right=True).clamp_(max=num_groups - 1)
    local = dst - padded_start[group]
    valid = (dst < padded_end[-1]) & (local < counts[group])
    src = (indptr[:-1][group] + local).clamp_(0, rows - 1)
    segmented = torch.where(
        valid.unsqueeze(1), linear[src], torch.zeros((), dtype=linear.dtype, device=linear.device)
    )
    return (
        segmented.reshape(cap // 128, 4, 32, cols // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .reshape(cap, cols)
        .contiguous()
        .view(a_descale.dtype)
    )


