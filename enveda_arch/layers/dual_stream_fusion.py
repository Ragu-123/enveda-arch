"""
Dual-Stream Conjugate Measure Fusion Layer
Couples direct fragment ion points {m_i} and conjugate neutral loss points {M_0 - m_i}
via bilateral cross-attention gating.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DualStreamConjugateFusion(nn.Module):
    """
    Bilateral cross-attention gating between Direct Fragment Ion Stream
    and Conjugate Neutral Loss Stream.
    """
    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        # Cross attention: Fragments query Losses
        self.cross_attn_frag_loss = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Cross attention: Losses query Fragments
        self.cross_attn_loss_frag = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # Gated fusion projection
        self.gate_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        frag_feats: torch.Tensor,
        loss_feats: torch.Tensor,
        mask: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Args:
            frag_feats: [B, N, D] features from direct fragment stream
            loss_feats: [B, N, D] features from conjugate neutral loss stream
            mask: [B, N] boolean mask (True for real peaks)
        Returns:
            fused_feats: [B, N, D] bilateral cross-gated features
        """
        # Multihead attention expects key_padding_mask where True indicates IGNORING the position
        key_padding_mask = ~mask if mask is not None else None

        # 1. Fragments query neutral losses (discovering which loss matches which fragment)
        frag_ctx, _ = self.cross_attn_frag_loss(
            query=frag_feats,
            key=loss_feats,
            value=loss_feats,
            key_padding_mask=key_padding_mask
        )

        # 2. Losses query fragments
        loss_ctx, _ = self.cross_attn_loss_frag(
            query=loss_feats,
            key=frag_feats,
            value=frag_feats,
            key_padding_mask=key_padding_mask
        )

        # Combine streams with adaptive gating
        combined_frag = frag_feats + frag_ctx
        combined_loss = loss_feats + loss_ctx

        concat_feats = torch.cat([combined_frag, combined_loss], dim=-1)
        gate = self.gate_proj(concat_feats)

        fused = gate * combined_frag + (1.0 - gate) * combined_loss
        fused = self.norm(fused)

        if mask is not None:
            fused = fused * mask.unsqueeze(-1)

        return fused
