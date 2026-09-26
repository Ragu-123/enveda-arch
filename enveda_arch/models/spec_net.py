"""
SpecContinuousNet: Complete Continuous Coordinate MS/MS Neural Network Architecture
Processes 1D continuous mass-spectral point clouds directly into multi-task chemical representations.
"""

import torch
import torch.nn as nn
from ..layers.fourier_coords import ContinuousFourierCoordinateEmbedding
from ..layers.continuous_conv import ContinuousFilterConvolutionBlock

class SpecContinuousNet(nn.Module):
    def __init__(
        self,
        d_model: int = 256,
        num_layers: int = 4,
        cond_dim: int = 3, # [precursor_mz, collision_energy, mode]
        fingerprint_dim: int = 2048,
        formula_dim: int = 8, # [C, H, N, O, P, S, F, Cl]
        sigma: float = 0.5
    ):
        super().__init__()
        self.d_model = d_model
        
        # Continuous Fourier Coordinate Embedders for fragments and neutral losses
        self.frag_fourier = ContinuousFourierCoordinateEmbedding(d_model=d_model // 2)
        self.loss_fourier = ContinuousFourierCoordinateEmbedding(d_model=d_model // 2)
        
        # Intensity projection (sqrt-transformed for dynamic range)
        self.input_proj = nn.Sequential(
            nn.Linear(d_model + 1, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU()
        )
        
        # Conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim, d_model // 2),
            nn.SiLU(),
            nn.Linear(d_model // 2, cond_dim)
        )
        
        # Stack of Continuous-Filter Convolution Blocks
        self.blocks = nn.ModuleList([
            ContinuousFilterConvolutionBlock(d_model=d_model, cond_dim=cond_dim, sigma=sigma * (2 ** i))
            for i in range(num_layers)
        ])
        
        # Permutation-Invariant Attention Pooling
        self.pool_attn = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )
        
        # Molecular Latent Embedding
        self.latent_proj = nn.Sequential(
            nn.Linear(d_model + cond_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )
        
        # Multi-Task Heads
        self.fingerprint_head = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, fingerprint_dim)
        )
        
        self.formula_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, formula_dim)
        )

    def forward(
        self,
        mzs: torch.Tensor,               # [B, P]
        intensities: torch.Tensor,       # [B, P]
        precursor_mz: torch.Tensor,      # [B, 1]
        collision_energy: torch.Tensor,  # [B, 1]
        mode: torch.Tensor,              # [B, 1]
        mask: torch.Tensor = None        # [B, P]
    ):
        B, P = mzs.shape
        
        # 1. Neutral loss point cloud
        neutral_losses = (precursor_mz - mzs).clamp(min=0.0) # [B, P]
        
        # 2. Continuous Fourier Coordinate Embeddings
        frag_emb = self.frag_fourier(mzs) # [B, P, d_model // 2]
        loss_emb = self.loss_fourier(neutral_losses) # [B, P, d_model // 2]
        
        # Square-root transformed peak intensities
        sqrt_ints = torch.sqrt(intensities.clamp(min=0.0)).unsqueeze(-1) # [B, P, 1]
        
        # Combined peak features
        peak_input = torch.cat([frag_emb, loss_emb, sqrt_ints], dim=-1) # [B, P, d_model + 1]
        x = self.input_proj(peak_input) # [B, P, d_model]
        
        # 3. Conditioning vector
        cond = torch.cat([precursor_mz / 500.0, collision_energy / 50.0, mode], dim=-1) # [B, 3]
        cond_feat = self.cond_proj(cond)
        
        # 4. Continuous-Filter Convolution Blocks
        for block in self.blocks:
            x = block(x, mzs, cond_feat, mask=mask)
            
        # 5. Permutation-Invariant Attention Pooling
        scores = self.pool_attn(x) # [B, P, 1]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(-1), -10000.0)
        weights = torch.softmax(scores, dim=1) # [B, P, 1]
        
        pooled = torch.sum(x * weights, dim=1) # [B, d_model]
        
        # Fuse with global precursor & energy context
        latent = self.latent_proj(torch.cat([pooled, cond_feat], dim=-1)) # [B, d_model]
        
        # 6. Heads
        fingerprint_logits = self.fingerprint_head(latent) # [B, fingerprint_dim]
        formula_preds = self.formula_head(latent)           # [B, formula_dim]
        
        return {
            "fingerprint_logits": fingerprint_logits,
            "formula_preds": formula_preds,
            "latent_embedding": latent
        }
