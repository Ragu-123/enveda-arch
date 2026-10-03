"""
Mathematical Triton Kernel Suite for MS/MS Fingerprint Learning
Contains:
1. PackedASLFunction / triton_packed_asl_loss:
   Fused Asymmetric Loss (Forward AND Backward autograd) directly over packed uint8 bit arrays (10,226 bits).
   Eliminates intermediate float target tensors and autograd activation tapes, cutting GPU VRAM bandwidth by 16x.
2. PackedSoftTanimotoFunction / triton_packed_soft_tanimoto_loss:
   Fused Soft Tanimoto Loss (Forward AND Backward autograd) directly over packed uint8 bit arrays.
   Computes exact analytical quotient-rule gradients in SRAM registers without memory traffic.
3. triton_fused_packed_dot_product:
   Direct dot-product between continuous query vectors and bit-packed candidates in register SRAM.
   Evaluates hundreds of thousands of candidates without unpacking to float in VRAM.
4. unpack_bits_torch:
   Vectorized tensor bit unpacking utility.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ==============================================================================
# 1. TRITON JIT KERNELS: FUSED BIT-PACKED ASYMMETRIC LOSS (FORWARD & BACKWARD)
# ==============================================================================
if HAS_TRITON:
    @triton.jit
    def _fused_packed_asl_fwd_kernel(
        logits_ptr,        # [B, N_BITS] float32/float16
        packed_ptr,        # [B, N_BYTES] uint8
        loss_out_ptr,      # [B] float32
        B, N_BITS, N_BYTES,
        stride_lb, stride_ln,
        stride_pb, stride_pn,
        gamma_pos: tl.constexpr,
        gamma_neg: tl.constexpr,
        margin: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        b_idx = tl.program_id(0)
        if b_idx >= B:
            return

        acc_loss = 0.0

        for offset in range(0, N_BITS, BLOCK_SIZE):
            bit_idx = offset + tl.arange(0, BLOCK_SIZE)
            mask = bit_idx < N_BITS

            # Load logit z
            l_ptrs = logits_ptr + b_idx * stride_lb + bit_idx * stride_ln
            z = tl.load(l_ptrs, mask=mask, other=0.0).to(tl.float32)

            # Map bit_idx to byte_idx and bit_shift
            byte_idx = bit_idx // 8
            bit_shift = bit_idx % 8

            p_ptrs = packed_ptr + b_idx * stride_pb + byte_idx * stride_pn
            packed_byte = tl.load(p_ptrs, mask=mask, other=0).to(tl.int32)

            # Extract target bit (0 or 1)
            target = (packed_byte >> bit_shift) & 1

            # Sigmoid probability: p = 1 / (1 + exp(-z))
            exp_neg_z = tl.exp(-tl.abs(z))
            p = tl.where(z >= 0.0, 1.0 / (1.0 + exp_neg_z), exp_neg_z / (1.0 + exp_neg_z))

            # 1. Positive term: L_pos = -(1 - p)^gamma_pos * log(clamp(p, eps, 1.0))
            p_clamp = tl.maximum(p, eps)
            one_minus_p = 1.0 - p
            w_pos = tl.exp(gamma_pos * tl.log(tl.maximum(one_minus_p, eps)))
            l_pos = -w_pos * tl.log(p_clamp)

            # 2. Negative term: p_m = max(p - margin, 0)
            p_m = tl.maximum(p - margin, 0.0)
            p_m_clamp = tl.maximum(p_m, eps)
            w_neg = tl.exp(gamma_neg * tl.log(p_m_clamp))
            l_neg = -w_neg * tl.log(tl.maximum(1.0 - p_m, eps))

            bit_loss = tl.where(target == 1, l_pos, l_neg)
            acc_loss += tl.sum(tl.where(mask, bit_loss, 0.0), axis=0)

        mean_loss = acc_loss / N_BITS
        tl.store(loss_out_ptr + b_idx, mean_loss)

    @triton.jit
    def _fused_packed_asl_bwd_kernel(
        logits_ptr,        # [B, N_BITS]
        packed_ptr,        # [B, N_BYTES]
        grad_out_ptr,      # [1] scalar loss gradient
        grad_logits_ptr,   # [B, N_BITS] output gradient wrt logits
        B, N_BITS, N_BYTES,
        stride_lb, stride_ln,
        stride_pb, stride_pn,
        gamma_pos: tl.constexpr,
        gamma_neg: tl.constexpr,
        margin: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        b_idx = tl.program_id(0)
        tile_idx = tl.program_id(1)
        if b_idx >= B:
            return

        bit_offset = tile_idx * BLOCK_SIZE
        bit_idx = bit_offset + tl.arange(0, BLOCK_SIZE)
        mask = bit_idx < N_BITS

        # Load grad_output scalar and normalize by (B * N_BITS)
        g_loss = tl.load(grad_out_ptr).to(tl.float32)
        scale = g_loss / (B * N_BITS)

        # Load logit z
        l_ptrs = logits_ptr + b_idx * stride_lb + bit_idx * stride_ln
        z = tl.load(l_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Extract target bit
        byte_idx = bit_idx // 8
        bit_shift = bit_idx % 8
        p_ptrs = packed_ptr + b_idx * stride_pb + byte_idx * stride_pn
        packed_byte = tl.load(p_ptrs, mask=mask, other=0).to(tl.int32)
        target = (packed_byte >> bit_shift) & 1

        # Sigmoid: p = 1 / (1 + exp(-z))
        exp_neg_z = tl.exp(-tl.abs(z))
        p = tl.where(z >= 0.0, 1.0 / (1.0 + exp_neg_z), exp_neg_z / (1.0 + exp_neg_z))
        one_minus_p = 1.0 - p

        # 1. Gradient for positive class: y = 1
        # dL/dz = (1 - p)^gamma_pos * [ gamma_pos * p * ln(p) - (1 - p) ]
        p_clamp = tl.maximum(p, eps)
        ln_p = tl.log(p_clamp)
        w_pos = tl.exp(gamma_pos * tl.log(tl.maximum(one_minus_p, eps)))
        g_pos = w_pos * (gamma_pos * p * ln_p - one_minus_p)

        # 2. Gradient for negative class: y = 0
        p_m = tl.maximum(p - margin, 0.0)
        p_m_clamp = tl.maximum(p_m, eps)
        one_minus_pm = tl.maximum(1.0 - p_m, eps)
        ln_one_minus_pm = tl.log(one_minus_pm)

        w_neg = tl.exp(gamma_neg * tl.log(p_m_clamp))
        w_neg_prev = tl.exp((gamma_neg - 1.0) * tl.log(p_m_clamp))
        g_neg_active = p * one_minus_p * (w_neg / one_minus_pm - gamma_neg * w_neg_prev * ln_one_minus_pm)
        g_neg = tl.where(p > margin, g_neg_active, 0.0)

        # Select gradient based on target bit
        grad_z = tl.where(target == 1, g_pos, g_neg) * scale

        # Store to grad_logits
        out_ptrs = grad_logits_ptr + b_idx * stride_lb + bit_idx * stride_ln
        tl.store(out_ptrs, grad_z, mask=mask)


class PackedASLFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, packed_targets, gamma_pos=0.0, gamma_neg=2.0, margin=0.05, eps=1e-7):
        B, N_BITS = logits.shape
        N_BYTES = packed_targets.shape[1]

        ctx.save_for_backward(logits, packed_targets)
        ctx.gamma_pos = float(gamma_pos)
        ctx.gamma_neg = float(gamma_neg)
        ctx.margin = float(margin)
        ctx.eps = float(eps)

        if not (HAS_TRITON and logits.is_cuda and packed_targets.is_cuda):
            shifts = torch.arange(8, device=logits.device, dtype=torch.uint8)
            unpacked = (packed_targets.unsqueeze(-1) >> shifts) & 1
            unpacked = unpacked.view(B, -1)[:, :N_BITS].float()

            p = torch.sigmoid(logits)
            loss_pos = - ((1.0 - p) ** gamma_pos) * torch.log(p.clamp(min=eps))
            p_m = (p - margin).clamp(min=0.0)
            loss_neg = - (p_m ** gamma_neg) * torch.log((1.0 - p_m).clamp(min=eps))
            loss = torch.where(unpacked == 1.0, loss_pos, loss_neg)
            return loss.mean()

        loss_out = torch.empty(B, device=logits.device, dtype=torch.float32)
        BLOCK_SIZE = 256
        _fused_packed_asl_fwd_kernel[(B,)](
            logits.contiguous(), packed_targets.contiguous(), loss_out,
            B, N_BITS, N_BYTES,
            logits.stride(0), logits.stride(1),
            packed_targets.stride(0), packed_targets.stride(1),
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            margin=margin,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return loss_out.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, packed_targets = ctx.saved_tensors
        B, N_BITS = logits.shape
        N_BYTES = packed_targets.shape[1]
        gamma_pos = ctx.gamma_pos
        gamma_neg = ctx.gamma_neg
        margin = ctx.margin
        eps = ctx.eps

        if not (HAS_TRITON and logits.is_cuda and packed_targets.is_cuda):
            shifts = torch.arange(8, device=logits.device, dtype=torch.uint8)
            unpacked = (packed_targets.unsqueeze(-1) >> shifts) & 1
            unpacked = unpacked.view(B, -1)[:, :N_BITS].float()

            p = torch.sigmoid(logits)
            one_minus_p = 1.0 - p
            scale = grad_output / (B * N_BITS)

            p_clamp = p.clamp(min=eps)
            w_pos = (one_minus_p.clamp(min=eps)) ** gamma_pos
            g_pos = w_pos * (gamma_pos * p * torch.log(p_clamp) - one_minus_p)

            p_m = (p - margin).clamp(min=0.0)
            p_m_clamp = p_m.clamp(min=eps)
            one_minus_pm = (1.0 - p_m).clamp(min=eps)
            w_neg = p_m_clamp ** gamma_neg
            w_neg_prev = p_m_clamp ** (gamma_neg - 1.0)
            g_neg_active = p * one_minus_p * (w_neg / one_minus_pm - gamma_neg * w_neg_prev * torch.log(one_minus_pm))
            g_neg = torch.where(p > margin, g_neg_active, torch.zeros_like(p))

            grad_logits = torch.where(unpacked == 1.0, g_pos, g_neg) * scale
            return grad_logits, None, None, None, None, None

        grad_logits = torch.empty_like(logits)
        BLOCK_SIZE = 256
        n_tiles = triton.cdiv(N_BITS, BLOCK_SIZE)
        grid = (B, n_tiles)

        grad_output_c = grad_output.contiguous()
        if grad_output_c.dim() == 0:
            grad_output_c = grad_output_c.view(1)

        _fused_packed_asl_bwd_kernel[grid](
            logits.contiguous(), packed_targets.contiguous(),
            grad_output_c, grad_logits,
            B, N_BITS, N_BYTES,
            logits.stride(0), logits.stride(1),
            packed_targets.stride(0), packed_targets.stride(1),
            gamma_pos=gamma_pos,
            gamma_neg=gamma_neg,
            margin=margin,
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE
        )
        return grad_logits, None, None, None, None, None


def triton_packed_asl_loss(
    logits: torch.Tensor,
    packed_targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 2.0,
    margin: float = 0.05,
    eps: float = 1e-7
) -> torch.Tensor:
    """
    Computes Asymmetric Loss with end-to-end autodiff directly from packed uint8 targets.
    """
    return PackedASLFunction.apply(logits, packed_targets, gamma_pos, gamma_neg, margin, eps)


# ==============================================================================
# 2. TRITON JIT KERNELS: FUSED SOFT TANIMOTO LOSS (FORWARD & BACKWARD)
# ==============================================================================
if HAS_TRITON:
    @triton.jit
    def _fused_packed_tanimoto_fwd_kernel(
        logits_ptr,        # [B, N_BITS]
        packed_ptr,        # [B, N_BYTES]
        n_out_ptr,         # [B] numerator N_b
        d_out_ptr,         # [B] denominator D_b
        loss_out_ptr,      # [B] loss L_b
        B, N_BITS, N_BYTES,
        stride_lb, stride_ln,
        stride_pb, stride_pn,
        eps: tl.constexpr,
        BLOCK_SIZE: tl.constexpr
    ):
        b_idx = tl.program_id(0)
        if b_idx >= B:
            return

        acc_num = 0.0
        acc_p = 0.0
        acc_y = 0.0

        for offset in range(0, N_BITS, BLOCK_SIZE):
            bit_idx = offset + tl.arange(0, BLOCK_SIZE)
            mask = bit_idx < N_BITS

            # Load z
            l_ptrs = logits_ptr + b_idx * stride_lb + bit_idx * stride_ln
            z = tl.load(l_ptrs, mask=mask, other=0.0).to(tl.float32)

            # Unpack target bit
            byte_idx = bit_idx // 8
            bit_shift = bit_idx % 8
            p_ptrs = packed_ptr + b_idx * stride_pb + byte_idx * stride_pn
            packed_byte = tl.load(p_ptrs, mask=mask, other=0).to(tl.int32)
            y = ((packed_byte >> bit_shift) & 1).to(tl.float32)

            # Sigmoid p = 1 / (1 + exp(-|z|))
            exp_neg_z = tl.exp(-tl.abs(z))
            p = tl.where(z >= 0.0, 1.0 / (1.0 + exp_neg_z), exp_neg_z / (1.0 + exp_neg_z))

            acc_num += tl.sum(tl.where(mask, p * y, 0.0), axis=0)
            acc_p += tl.sum(tl.where(mask, p, 0.0), axis=0)
            acc_y += tl.sum(tl.where(mask, y, 0.0), axis=0)

        den = acc_p + acc_y - acc_num + eps
        tani = acc_num / den
        loss = 1.0 - tani

        tl.store(n_out_ptr + b_idx, acc_num)
        tl.store(d_out_ptr + b_idx, den)
        tl.store(loss_out_ptr + b_idx, loss)

    @triton.jit
    def _fused_packed_tanimoto_bwd_kernel(
        logits_ptr,        # [B, N_BITS]
        packed_ptr,        # [B, N_BYTES]
        n_in_ptr,          # [B]
        d_in_ptr,          # [B]
        grad_out_ptr,      # [1]
        grad_logits_ptr,   # [B, N_BITS]
        B, N_BITS, N_BYTES,
        stride_lb, stride_ln,
        stride_pb, stride_pn,
        BLOCK_SIZE: tl.constexpr
    ):
        b_idx = tl.program_id(0)
        tile_idx = tl.program_id(1)
        if b_idx >= B:
            return

        bit_offset = tile_idx * BLOCK_SIZE
        bit_idx = bit_offset + tl.arange(0, BLOCK_SIZE)
        mask = bit_idx < N_BITS

        # Load scalar N_b, D_b, and grad_out
        nb = tl.load(n_in_ptr + b_idx).to(tl.float32)
        db = tl.load(d_in_ptr + b_idx).to(tl.float32)
        g_loss = tl.load(grad_out_ptr).to(tl.float32)
        scale = g_loss / B

        # Load z
        l_ptrs = logits_ptr + b_idx * stride_lb + bit_idx * stride_ln
        z = tl.load(l_ptrs, mask=mask, other=0.0).to(tl.float32)

        # Unpack y
        byte_idx = bit_idx // 8
        bit_shift = bit_idx % 8
        p_ptrs = packed_ptr + b_idx * stride_pb + byte_idx * stride_pn
        packed_byte = tl.load(p_ptrs, mask=mask, other=0).to(tl.int32)
        y = ((packed_byte >> bit_shift) & 1).to(tl.float32)

        # Sigmoid
        exp_neg_z = tl.exp(-tl.abs(z))
        p = tl.where(z >= 0.0, 1.0 / (1.0 + exp_neg_z), exp_neg_z / (1.0 + exp_neg_z))

        # Analytical derivative: dL_b / dp = (N_b - y*(D_b + N_b)) / (D_b^2)
        # dL_b / dz = (dL_b / dp) * p * (1 - p)
        d_sq = db * db
        dp = (nb - y * (db + nb)) / d_sq
        dz = dp * p * (1.0 - p) * scale

        out_ptrs = grad_logits_ptr + b_idx * stride_lb + bit_idx * stride_ln
        tl.store(out_ptrs, dz, mask=mask)


class PackedSoftTanimotoFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, packed_targets, eps=1e-7):
        B, N_BITS = logits.shape
        N_BYTES = packed_targets.shape[1]

        if not (HAS_TRITON and logits.is_cuda and packed_targets.is_cuda):
            shifts = torch.arange(8, device=logits.device, dtype=torch.uint8)
            unpacked = (packed_targets.unsqueeze(-1) >> shifts) & 1
            unpacked = unpacked.view(B, -1)[:, :N_BITS].float()
            p = torch.sigmoid(logits)
            num = (p * unpacked).sum(dim=-1)
            den = p.sum(dim=-1) + unpacked.sum(dim=-1) - num + eps
            return (1.0 - (num / den)).mean()

        n_out = torch.empty(B, device=logits.device, dtype=torch.float32)
        d_out = torch.empty(B, device=logits.device, dtype=torch.float32)
        loss_out = torch.empty(B, device=logits.device, dtype=torch.float32)

        BLOCK_SIZE = 256
        _fused_packed_tanimoto_fwd_kernel[(B,)](
            logits.contiguous(), packed_targets.contiguous(),
            n_out, d_out, loss_out,
            B, N_BITS, N_BYTES,
            logits.stride(0), logits.stride(1),
            packed_targets.stride(0), packed_targets.stride(1),
            eps=eps,
            BLOCK_SIZE=BLOCK_SIZE
        )

        ctx.save_for_backward(logits, packed_targets, n_out, d_out)
        return loss_out.mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, packed_targets, n_out, d_out = ctx.saved_tensors
        B, N_BITS = logits.shape
        N_BYTES = packed_targets.shape[1]

        if not (HAS_TRITON and logits.is_cuda and packed_targets.is_cuda):
            shifts = torch.arange(8, device=logits.device, dtype=torch.uint8)
            unpacked = (packed_targets.unsqueeze(-1) >> shifts) & 1
            unpacked = unpacked.view(B, -1)[:, :N_BITS].float()
            p = torch.sigmoid(logits)
            num = (p * unpacked).sum(dim=-1, keepdim=True)
            den = (p.sum(dim=-1, keepdim=True) + unpacked.sum(dim=-1, keepdim=True) - num + 1e-7)
            dp = (num - unpacked * (den + num)) / (den ** 2)
            scale = grad_output / B
            grad_logits = dp * p * (1.0 - p) * scale
            return grad_logits, None, None

        grad_logits = torch.empty_like(logits)
        BLOCK_SIZE = 256
        grid = (B, triton.cdiv(N_BITS, BLOCK_SIZE))

        grad_output_c = grad_output.contiguous()
        if grad_output_c.dim() == 0:
            grad_output_c = grad_output_c.view(1)

        _fused_packed_tanimoto_bwd_kernel[grid](
            logits.contiguous(), packed_targets.contiguous(),
            n_out, d_out, grad_output_c, grad_logits,
            B, N_BITS, N_BYTES,
            logits.stride(0), logits.stride(1),
            packed_targets.stride(0), packed_targets.stride(1),
            BLOCK_SIZE=BLOCK_SIZE
        )
        return grad_logits, None, None


def triton_packed_soft_tanimoto_loss(logits: torch.Tensor, packed_targets: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Computes Soft Tanimoto IoU Loss with analytical autograd directly from packed uint8 targets.
    """
    return PackedSoftTanimotoFunction.apply(logits, packed_targets, eps)


# ==============================================================================
# 3. TRITON FUSED PACKED BIT MATRIX DOT PRODUCT (FOR FAST CANDIDATE SCORING)
# ==============================================================================
if HAS_TRITON:
    @triton.jit
    def _fused_packed_dot_kernel(
        query_ptr,       # [B, N_BITS] float32 query vectors
        cand_ptr,        # [K, N_BYTES] uint8 candidate fingerprints
        scores_ptr,      # [B, K] float32 output dot product scores
        B, K, N_BITS, N_BYTES,
        stride_qb, stride_qn,
        stride_ck, stride_cn,
        stride_sb, stride_sk,
        BLOCK_K: tl.constexpr,
        BLOCK_BITS: tl.constexpr
    ):
        b_idx = tl.program_id(0)
        k_block_idx = tl.program_id(1)

        if b_idx >= B:
            return

        k_idx = k_block_idx * BLOCK_K + tl.arange(0, BLOCK_K)
        k_mask = k_idx < K

        acc = tl.zeros([BLOCK_K], dtype=tl.float32)

        for bit_offset in range(0, N_BITS, BLOCK_BITS):
            bit_local = tl.arange(0, BLOCK_BITS)
            bit_global = bit_offset + bit_local
            bit_mask = bit_global < N_BITS

            # Load query vector slice: [BLOCK_BITS]
            q_ptrs = query_ptr + b_idx * stride_qb + bit_global * stride_qn
            q_vals = tl.load(q_ptrs, mask=bit_mask, other=0.0).to(tl.float32)

            # For each bit, find byte index and shift
            byte_global = bit_global // 8
            bit_shift = bit_global % 8

            # Load candidate bytes: [BLOCK_K, BLOCK_BITS]
            c_ptrs = cand_ptr + k_idx[:, None] * stride_ck + byte_global[None, :] * stride_cn
            c_bytes = tl.load(c_ptrs, mask=(k_mask[:, None] & bit_mask[None, :]), other=0).to(tl.int32)

            # Extract bits: [BLOCK_K, BLOCK_BITS]
            c_bits = ((c_bytes >> bit_shift[None, :]) & 1).to(tl.float32)

            # Multiply and accumulate across bits
            acc += tl.sum(c_bits * q_vals[None, :], axis=1)

        # Store to scores [B, K]
        s_ptrs = scores_ptr + b_idx * stride_sb + k_idx * stride_sk
        tl.store(s_ptrs, acc, mask=k_mask)


def triton_fused_packed_dot_product(
    queries: torch.Tensor,       # [B, N_BITS] float32
    candidates: torch.Tensor,    # [K, N_BYTES] uint8
    block_k: int = 64
) -> torch.Tensor:
    """
    Computes query-candidate dot products directly from packed candidate bit arrays in SRAM.
    Scores K candidates against B queries without materializing float candidate matrix.
    """
    B, N_BITS = queries.shape
    K, N_BYTES = candidates.shape

    if not (HAS_TRITON and queries.is_cuda and candidates.is_cuda):
        cand_unpacked = unpack_bits_torch(candidates, n_bits=N_BITS)
        return torch.matmul(queries, cand_unpacked.t())

    scores = torch.empty((B, K), device=queries.device, dtype=torch.float32)
    BLOCK_BITS = 128
    grid = (B, triton.cdiv(K, block_k))

    _fused_packed_dot_kernel[grid](
        queries.contiguous(), candidates.contiguous(), scores,
        B, K, N_BITS, N_BYTES,
        queries.stride(0), queries.stride(1),
        candidates.stride(0), candidates.stride(1),
        scores.stride(0), scores.stride(1),
        BLOCK_K=block_k,
        BLOCK_BITS=BLOCK_BITS
    )
    return scores


def unpack_bits_torch(packed_bytes: torch.Tensor, n_bits: int = 10226) -> torch.Tensor:
    """Unpacks [K, N_BYTES] uint8 into [K, N_BITS] float32 tensor."""
    shifts = torch.arange(8, device=packed_bytes.device, dtype=torch.uint8)
    unpacked = (packed_bytes.unsqueeze(-1) >> shifts) & 1
    return unpacked.view(packed_bytes.shape[0], -1)[:, :n_bits].float()
