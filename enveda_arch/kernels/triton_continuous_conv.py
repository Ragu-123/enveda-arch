"""
Custom Triton Kernel for Continuous Coordinate Gaussian-RBF Convolution
Computes pairwise continuous coordinate interactions in GPU registers with O(P) SRAM usage.
"""

import torch
import math

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


if HAS_TRITON:
    @triton.jit
    def _triton_continuous_rbf_conv_kernel(
        x_ptr,           # [B, P, D] input features
        m_ptr,           # [B, P] continuous coordinates (masses)
        out_ptr,         # [B, P, D] output features
        sigma_sq,        # 2 * sigma^2 constant
        B, P, D,         # Dimensions
        stride_xb, stride_xp, stride_xd,
        stride_mb, stride_mp,
        stride_ob, stride_op, stride_od,
        BLOCK_P: tl.constexpr,
        BLOCK_D: tl.constexpr
    ):
        """
        Fused kernel: out[b, i, d] = sum_j x[b, j, d] * exp(-(m[b, i] - m[b, j])^2 / (2 * sigma^2))
        """
        pid_b = tl.program_id(0) # Batch index
        pid_i = tl.program_id(1) # Target peak index

        if pid_b >= B or pid_i >= P:
            return

        # Load target coordinate m[b, i]
        m_i = tl.load(m_ptr + pid_b * stride_mb + pid_i * stride_mp)

        # Iterate over feature dimensions in chunks of BLOCK_D
        for d_offset in range(0, D, BLOCK_D):
            d_idx = d_offset + tl.arange(0, BLOCK_D)
            d_mask = d_idx < D

            acc = tl.zeros([BLOCK_D], dtype=tl.float32)

            # Accumulate over all source peaks j in chunks of BLOCK_P
            for j_offset in range(0, P, BLOCK_P):
                j_idx = j_offset + tl.arange(0, BLOCK_P)
                j_mask = j_idx < P

                # Load source coordinates m[b, j]
                m_j = tl.load(m_ptr + pid_b * stride_mb + j_idx * stride_mp, mask=j_mask, other=0.0)

                # Compute continuous coordinate distance squared
                diff = m_i - m_j
                dist_sq = diff * diff
                weight = tl.exp(-dist_sq / sigma_sq) # Shape [BLOCK_P]

                # Load x[b, j, d]
                # x slice shape: [BLOCK_P, BLOCK_D]
                x_ptrs = x_ptr + pid_b * stride_xb + j_idx[:, None] * stride_xp + d_idx[None, :] * stride_xd
                x_vals = tl.load(x_ptrs, mask=(j_mask[:, None] & d_mask[None, :]), other=0.0)

                # Multiply by RBF weight and sum over j
                weighted_x = x_vals * weight[:, None]
                acc += tl.sum(weighted_x, axis=0)

            # Store result
            out_ptrs = out_ptr + pid_b * stride_ob + pid_i * stride_op + d_idx * stride_od
            tl.store(out_ptrs, acc, mask=d_mask)


def triton_continuous_rbf_conv(x: torch.Tensor, m: torch.Tensor, sigma: float = 0.5) -> torch.Tensor:
    """
    Continuous RBF Convolution using Triton if available, falling back to PyTorch.
    Args:
        x: [B, P, D] feature tensor
        m: [B, P] continuous coordinates
        sigma: bandwidth of Gaussian RBF kernel
    Returns:
        out: [B, P, D] aggregated features
    """
    if not (HAS_TRITON and x.is_cuda and x.dtype == torch.float32):
        # High-performance PyTorch fallback
        diff = m.unsqueeze(2) - m.unsqueeze(1) # [B, P, P]
        weights = torch.exp(- (diff ** 2) / (2.0 * sigma ** 2)) # [B, P, P]
        return torch.bmm(weights, x) # [B, P, D]

    B, P, D = x.shape
    out = torch.empty_like(x)
    sigma_sq = 2.0 * (sigma ** 2)

    BLOCK_P = 64 if P >= 64 else 32
    BLOCK_D = 32 if D >= 32 else 16

    grid = (B, P)
    _triton_continuous_rbf_conv_kernel[grid](
        x, m, out,
        sigma_sq,
        B, P, D,
        x.stride(0), x.stride(1), x.stride(2),
        m.stride(0), m.stride(1),
        out.stride(0), out.stride(1), out.stride(2),
        BLOCK_P=BLOCK_P,
        BLOCK_D=BLOCK_D
    )
    return out
