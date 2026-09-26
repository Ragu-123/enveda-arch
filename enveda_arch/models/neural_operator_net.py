"""
SpecNeuralOperatorNet: Multiscale Continuous Kernel Neural Operator for Tandem Mass Spectrometry
Synthesized from:
- Kovachki, Stuart et al. (Neural Operator on Function Spaces, 2021)
- Mallat (Invariant Scattering Representations & Multiscale Wavelets, 2011)
- Bochner's Theorem (Random Fourier Features on Continuous Metric Spaces)
- De Waele et al. (Bayes Retrieval Regret Bounds & Contrastive Embedding Spaces, ICML 2026)
"""

import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from enveda_arch.layers.fourier_coords import RandomFourierFeatures
from enveda_arch.layers.multiscale_operator import MultiscaleContinuousKernelOperator
from enveda_arch.layers.dual_stream_fusion import DualStreamConjugateFusion
from enveda_arch.layers.film_energy_modulator import FiLMEnergyModulator

class SpecNeuralOperatorNet(nn.Module):
    """
    Multiscale Continuous Neural Operator with Dual-Stream Conjugate Measure Geometry.
    """
    def __init__(
        self,
        hidden_dim: int = 256,
        retrieval_dim: int = 256,
        fingerprint_dim: int = 2048,
        formula_dim: int = 10,
        num_operator_layers: int = 2,
        sigmas: Tuple[float, ...] = (0.02, 0.5, 5.0, 28.0),
        num_heads: int = 4,
        fourier_dim: int = 128
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.retrieval_dim = retrieval_dim
        self.fingerprint_dim = fingerprint_dim
        self.formula_dim = formula_dim

        # 1. Continuous Coordinate Harmonic Encoders (Bochner's Theorem)
        self.rff_frag = RandomFourierFeatures(fourier_dim=fourier_dim, sigma=1.0)
        self.rff_loss = RandomFourierFeatures(fourier_dim=fourier_dim, sigma=1.0)

        # Peak feature projections: [fourier_dim + 1 (intensity)] -> hidden_dim
        self.frag_proj = nn.Linear(fourier_dim + 1, hidden_dim)
        self.loss_proj = nn.Linear(fourier_dim + 1, hidden_dim)

        # 2. Multiscale Continuous Kernel Operator Layers
        self.frag_operators = nn.ModuleList([
            MultiscaleContinuousKernelOperator(hidden_dim, hidden_dim, sigmas=sigmas)
            for _ in range(num_operator_layers)
        ])
        self.loss_operators = nn.ModuleList([
            MultiscaleContinuousKernelOperator(hidden_dim, hidden_dim, sigmas=sigmas)
            for _ in range(num_operator_layers)
        ])

        # 3. Dual-Stream Conjugate Bilateral Cross-Gated Fusion
        self.conjugate_fusion = DualStreamConjugateFusion(hidden_dim, num_heads=num_heads)

        # 4. Thermodynamic State FiLM Modulator (Collision Energy + Precursor Mass + Ionization Mode)
        self.film_modulator = FiLMEnergyModulator(cond_dim=3, d_model=hidden_dim)

        # 5. Permutation-Invariant Attention Pooling (Set Operator)
        self.pool_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.pool_attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.pool_norm = nn.LayerNorm(hidden_dim)

        # 6. Multi-Task Output Projections
        # Head A: Molecular Fingerprint Logits (ASL + Soft Tanimoto)
        self.head_fingerprint = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, fingerprint_dim)
        )

        # Head B: Direct Metric Retrieval Projection (De Waele et al. 2026 ICML)
        self.head_retrieval = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, retrieval_dim)
        )

        # Candidate Fingerprint Metric Projection (maps candidate fingerprints to retrieval space)
        self.candidate_proj = nn.Sequential(
            nn.Linear(fingerprint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, retrieval_dim)
        )

        # Head C: Molecular Formula Elemental Atom Counts
        self.head_formula = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, formula_dim)
        )

    def project_candidate_fingerprint(self, cand_fps: torch.Tensor) -> torch.Tensor:
        """
        Projects candidate binary fingerprint vectors into L2-normalized metric retrieval space.
        Args:
            cand_fps: [B, fingerprint_dim] or [N_cand, fingerprint_dim]
        Returns:
            cand_embeddings: [..., retrieval_dim]
        """
        z_cand = self.candidate_proj(cand_fps)
        return F.normalize(z_cand, p=2, dim=-1)

    def forward(
        self,
        mzs: torch.Tensor,
        intensities: torch.Tensor,
        precursor_mz: torch.Tensor,
        collision_energy: torch.Tensor,
        mode: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward evaluation of the continuous neural operator.
        """
        B, N = mzs.shape

        if mask is None:
            mask = mzs > 0.0

        # Compute conjugate neutral losses: Delta m_i = M_0 - m_i
        neutral_losses = torch.clamp(precursor_mz.view(B, 1) - mzs, min=0.0)

        # Continuous harmonic coordinate representations
        phi_frag = self.rff_frag(mzs)
        phi_loss = self.rff_loss(neutral_losses)

        # Append intensities
        x_frag = torch.cat([phi_frag, intensities.unsqueeze(-1)], dim=-1)
        x_loss = torch.cat([phi_loss, intensities.unsqueeze(-1)], dim=-1)

        h_frag = self.frag_proj(x_frag)
        h_loss = self.loss_proj(x_loss)

        # Apply Multiscale Continuous Kernel Operators along both streams
        for op_frag, op_loss in zip(self.frag_operators, self.loss_operators):
            h_frag = op_frag(h_frag, mzs, intensities=intensities, mask=mask)
            h_loss = op_loss(h_loss, neutral_losses, intensities=intensities, mask=mask)

        # Dual-Stream Conjugate Bilateral Cross-Gated Fusion
        h_fused = self.conjugate_fusion(h_frag, h_loss, mask=mask)

        # Apply FiLM Energy Modulation
        cond = torch.cat([precursor_mz / 1000.0, collision_energy / 100.0, mode], dim=-1)
        h_fused = self.film_modulator(h_fused, cond)

        # Permutation-Invariant Attention Pooling
        pool_q = self.pool_query.expand(B, -1, -1)
        key_padding_mask = ~mask if mask is not None else None

        pooled, _ = self.pool_attn(
            query=pool_q,
            key=h_fused,
            value=h_fused,
            key_padding_mask=key_padding_mask
        )
        h_spec = self.pool_norm(pooled.squeeze(1))

        # Output predictions
        fp_logits = self.head_fingerprint(h_spec)
        ret_embed = F.normalize(self.head_retrieval(h_spec), p=2, dim=-1)
        form_preds = self.head_formula(h_spec)

        return {
            "fingerprint_logits": fp_logits,
            "retrieval_embedding": ret_embed,
            "formula_preds": form_preds,
            "spectrum_embedding": h_spec
        }
