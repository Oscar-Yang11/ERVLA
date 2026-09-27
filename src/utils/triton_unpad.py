import torch
import triton
import triton.language as tl


# ─────────────────────────── Forward ───────────────────────────


@triton.jit
def _unpad_all_kernel(
    k_src, k_dst,
    v_src, v_dst,
    pid_src, pid_dst,
    starts_ptr, seq_len_ptr,
    k_s0, k_s1, k_s2, k_s3, k_s4,
    p_s0, p_s1, p_s2,
    dk_s0, dk_s1, dk_s2, dk_s3, dk_s4,
    dp_s0, dp_s1, dp_s2,
    num_layers, num_heads, head_dim,
    LH,
    max_len,
    num_tiles,
    BLOCK_HEAD: tl.constexpr,
    TILE: tl.constexpr,
):
    """Gather: src[start[b]+j] → dst[b, j]"""
    pid = tl.program_id(0)
    b = pid // num_tiles
    tile_idx = pid - b * num_tiles
    sl = tl.load(seq_len_ptr + b)
    start = tl.load(starts_ptr + b)
    j_base = tile_idx * TILE

    for t in range(TILE):
        j = j_base + t
        if j < sl:
            src_pos = start + j
            dst_pos = j
            for lh in range(LH):
                l = lh // num_heads
                h = lh - l * num_heads
                o = tl.arange(0, BLOCK_HEAD)
                m = o < head_dim
                tl.store(
                    k_dst + l * dk_s0 + b * dk_s1 + h * dk_s2 + dst_pos * dk_s3 + o,
                    tl.load(k_src + l * k_s0 + h * k_s2 + src_pos * k_s3 + o, mask=m),
                    mask=m,
                )
                tl.store(
                    v_dst + l * dk_s0 + b * dk_s1 + h * dk_s2 + dst_pos * dk_s3 + o,
                    tl.load(v_src + l * k_s0 + h * k_s2 + src_pos * k_s3 + o, mask=m),
                    mask=m,
                )
            for p in range(3):
                tl.store(
                    pid_dst + p * dp_s0 + b * dp_s1 + dst_pos * dp_s2,
                    tl.load(pid_src + p * p_s0 + src_pos * p_s2),
                )


# ─────────────────────────── Backward ───────────────────────────


@triton.jit
def _unpad_backward_kernel(
    grad_k_dst, grad_k_src,
    grad_v_dst, grad_v_src,
    starts_ptr, seq_len_ptr,
    # grad_dst strides: (num_layers, action_bs, num_heads, max_length, head_dim)
    gd_s0, gd_s1, gd_s2, gd_s3, gd_s4,
    # grad_src strides: (num_layers, 1, num_heads, total_seq, head_dim)
    gs_s0, gs_s1, gs_s2, gs_s3, gs_s4,
    num_layers, num_heads, head_dim,
    LH,
    max_len,
    num_tiles,
    BLOCK_HEAD: tl.constexpr,
    TILE: tl.constexpr,
):
    """Scatter: grad_dst[b, j] → grad_src[start[b]+j] (no atomic needed, disjoint writes)"""
    pid = tl.program_id(0)
    b = pid // num_tiles
    tile_idx = pid - b * num_tiles
    sl = tl.load(seq_len_ptr + b)
    start = tl.load(starts_ptr + b)
    j_base = tile_idx * TILE

    for t in range(TILE):
        j = j_base + t
        if j < sl:
            src_pos = start + j
            dst_pos = j
            for lh in range(LH):
                l = lh // num_heads
                h = lh - l * num_heads
                o = tl.arange(0, BLOCK_HEAD)
                m = o < head_dim
                tl.store(
                    grad_k_src + l * gs_s0 + h * gs_s2 + src_pos * gs_s3 + o,
                    tl.load(grad_k_dst + l * gd_s0 + b * gd_s1 + h * gd_s2 + dst_pos * gd_s3 + o, mask=m),
                    mask=m,
                )
                tl.store(
                    grad_v_src + l * gs_s0 + h * gs_s2 + src_pos * gs_s3 + o,
                    tl.load(grad_v_dst + l * gd_s0 + b * gd_s1 + h * gd_s2 + dst_pos * gd_s3 + o, mask=m),
                    mask=m,
                )


# ─────────────────────────── Autograd Function ───────────────────────────


def _next_pow2(x: int) -> int:
    return 1 << (x - 1).bit_length()


class UnpadAllTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, all_keys, all_values, position_ids,
                starts, seq_len, action_bs, max_length):
        num_layers = all_keys.shape[0]
        num_heads = all_keys.shape[2]
        head_dim = all_keys.shape[4]
        LH = num_layers * num_heads

        device = all_keys.device
        starts_i = starts.to(device, non_blocking=True).int()
        seq_len_i = seq_len.to(device, non_blocking=True).int()

        new_keys = torch.zeros(num_layers, action_bs, num_heads, max_length, head_dim,
                               device=device, dtype=all_keys.dtype)
        new_values = torch.zeros_like(new_keys)
        new_position_ids = torch.zeros(3, action_bs, max_length,
                                       device=device, dtype=position_ids.dtype)

        BLOCK_HEAD = _next_pow2(head_dim)
        TILE = 1
        num_tiles = (max_length + TILE - 1) // TILE

        _unpad_all_kernel[(action_bs * num_tiles,)](
            all_keys, new_keys,
            all_values, new_values,
            position_ids, new_position_ids,
            starts_i, seq_len_i,
            all_keys.stride(0), all_keys.stride(1), all_keys.stride(2),
            all_keys.stride(3), all_keys.stride(4),
            position_ids.stride(0), position_ids.stride(1), position_ids.stride(2),
            new_keys.stride(0), new_keys.stride(1), new_keys.stride(2),
            new_keys.stride(3), new_keys.stride(4),
            new_position_ids.stride(0), new_position_ids.stride(1), new_position_ids.stride(2),
            num_layers, num_heads, head_dim, LH,
            int(max_length), int(num_tiles),
            BLOCK_HEAD=BLOCK_HEAD, TILE=TILE,
        )

        ctx.save_for_backward(starts_i, seq_len_i)
        ctx.src_shape = all_keys.shape  # (num_layers, 1, num_heads, total_seq, head_dim)
        ctx.dst_strides = (new_keys.stride(0), new_keys.stride(1), new_keys.stride(2),
                           new_keys.stride(3), new_keys.stride(4))
        ctx.meta = (num_layers, num_heads, head_dim, LH, max_length, num_tiles, BLOCK_HEAD, TILE)

        return new_keys, new_values, new_position_ids

    @staticmethod
    def backward(ctx, grad_keys, grad_values, grad_position_ids):
        starts_i, seq_len_i = ctx.saved_tensors
        num_layers, num_heads, head_dim, LH, max_length, num_tiles, BLOCK_HEAD, TILE = ctx.meta
        src_shape = ctx.src_shape
        gd_s0, gd_s1, gd_s2, gd_s3, gd_s4 = ctx.dst_strides

        device = grad_keys.device
        action_bs = grad_keys.shape[1]

        grad_k_src = torch.zeros(*src_shape, device=device, dtype=grad_keys.dtype)
        grad_v_src = torch.zeros(*src_shape, device=device, dtype=grad_values.dtype)

        _unpad_backward_kernel[(action_bs * num_tiles,)](
            grad_keys.contiguous(), grad_k_src,
            grad_values.contiguous(), grad_v_src,
            starts_i, seq_len_i,
            gd_s0, gd_s1, gd_s2, gd_s3, gd_s4,
            grad_k_src.stride(0), grad_k_src.stride(1), grad_k_src.stride(2),
            grad_k_src.stride(3), grad_k_src.stride(4),
            num_layers, num_heads, head_dim, LH,
            int(max_length), int(num_tiles),
            BLOCK_HEAD=BLOCK_HEAD, TILE=TILE,
        )

        # all_keys, all_values, position_ids, starts, seq_len, action_bs, max_length
        return grad_k_src, grad_v_src, None, None, None, None, None


def unpad_all_triton(all_keys, all_values, position_ids,
                     starts, seq_len, action_bs, max_length,
                     TILE=1):
    """Unpad K, V, position_ids with autograd support.

    Args:
        all_keys:    (num_layers, 1, num_heads, total_seq, head_dim)
        all_values:  (num_layers, 1, num_heads, total_seq, head_dim)
        position_ids: (3, 1, total_seq)
        starts:      (action_bs,)
        seq_len:     (action_bs,)
    Returns:
        new_keys, new_values: (num_layers, action_bs, num_heads, max_length, head_dim)
        new_position_ids: (3, action_bs, max_length)
    """
    return UnpadAllTriton.apply(all_keys, all_values, position_ids,
                                starts, seq_len, action_bs, max_length)
