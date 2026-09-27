"""
Triton fused kernel for Vision RoPE in Qwen3-VL.

Input layout:
  q, k : [seq, n_head, head_dim]   bf16
  cos  : [seq, head_dim]           bf16  (left half = right half, standard RoPE)
  sin  : [seq, head_dim]           bf16

Forward:  x_out = x * cos + rotate_half(x) * sin
Backward: dx    = dx_out * cos - rotate_half(dx_out) * sin
          (equivalent to applying RoPE with -sin, i.e. same kernel with sin negated)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _vision_rope_kernel(
    q_ptr, k_ptr,
    cos_ptr, sin_ptr,
    seq_len,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    head_dim: tl.constexpr,
    half_dim: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    # grid: (seq_len,)  — one program per token
    token_id = tl.program_id(0)

    # Load cos/sin for this token: shape [head_dim]
    # Only need first half_dim values (left half = right half in standard RoPE)
    cos_offsets = tl.arange(0, half_dim)
    cos_row = tl.load(cos_ptr + token_id * head_dim + cos_offsets).to(tl.float32)
    sin_row = tl.load(sin_ptr + token_id * head_dim + cos_offsets).to(tl.float32)

    if BACKWARD_PASS:
        sin_row = -sin_row

    # Process q heads
    for h in tl.static_range(n_qh):
        base = q_ptr + token_id * n_qh * head_dim + h * head_dim
        x1_offsets = tl.arange(0, half_dim)
        x2_offsets = tl.arange(0, half_dim) + half_dim

        x1 = tl.load(base + x1_offsets).to(tl.float32)
        x2 = tl.load(base + x2_offsets).to(tl.float32)

        new_x1 = x1 * cos_row - x2 * sin_row
        new_x2 = x2 * cos_row + x1 * sin_row

        tl.store(base + x1_offsets, new_x1.to(tl.bfloat16))
        tl.store(base + x2_offsets, new_x2.to(tl.bfloat16))

    # Process k heads
    for h in tl.static_range(n_kh):
        base = k_ptr + token_id * n_kh * head_dim + h * head_dim
        x1_offsets = tl.arange(0, half_dim)
        x2_offsets = tl.arange(0, half_dim) + half_dim

        x1 = tl.load(base + x1_offsets).to(tl.float32)
        x2 = tl.load(base + x2_offsets).to(tl.float32)

        new_x1 = x1 * cos_row - x2 * sin_row
        new_x2 = x2 * cos_row + x1 * sin_row

        tl.store(base + x1_offsets, new_x1.to(tl.bfloat16))
        tl.store(base + x2_offsets, new_x2.to(tl.bfloat16))


class VisionRopeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, cos, sin):
        # q, k: [seq, n_head, head_dim] bf16, contiguous
        # cos, sin: [seq, head_dim] bf16
        seq_len, n_qh, head_dim = q.shape
        n_kh = k.shape[1]
        half_dim = head_dim // 2

        q = q.contiguous()
        k = k.contiguous()
        cos = cos.contiguous()
        sin = sin.contiguous()

        _vision_rope_kernel[(seq_len,)](
            q, k, cos, sin,
            seq_len,
            n_qh=n_qh, n_kh=n_kh,
            head_dim=head_dim, half_dim=half_dim,
            BACKWARD_PASS=False,
        )
        ctx.save_for_backward(cos, sin)
        return q, k

    @staticmethod
    def backward(ctx, dq, dk):
        cos, sin = ctx.saved_tensors
        seq_len, n_qh, head_dim = dq.shape
        n_kh = dk.shape[1]
        half_dim = head_dim // 2

        dq = dq.contiguous()
        dk = dk.contiguous()

        _vision_rope_kernel[(seq_len,)](
            dq, dk, cos, sin,
            seq_len,
            n_qh=n_qh, n_kh=n_kh,
            head_dim=head_dim, half_dim=half_dim,
            BACKWARD_PASS=True,
        )
        return dq, dk, None, None


def vision_rope_forward(q, k, cos, sin):
    """
    Apply Vision RoPE inplace.
    q, k: [seq, n_head, head_dim] bf16
    cos, sin: [seq, head_dim] bf16  (standard RoPE: left half = right half)
    """
    return VisionRopeFunction.apply(q, k, cos, sin)
