"""
Listwise Plackett-Luce Isomer Ranking Loss (De Waele et al., ICML 2026)
Directly optimizes reciprocal rank of true candidate relative to mass-matched constitutional isomers (+-10 ppm).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class PlackettLuceIsomerLoss(nn.Module):
    """
    Listwise Plackett-Luce Ranking Loss over mass-matched constitutional isomers.
    For each spectrum in the batch, contrasts score of true candidate s_true
    against scores of K mass-matched isomers s_neg_1, ..., s_neg_K.
    """
    def __init__(self, temperature: float = 0.10):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        query_vec: torch.Tensor,       # [B, D] (retrieval embedding or logits)
        true_cand_vec: torch.Tensor,   # [B, D] (true candidate representation)
        isomer_cand_vecs: torch.Tensor # [B, K, D] (mass-matched isomer representations)
    ) -> torch.Tensor:
        """
        Args:
            query_vec: [B, D]
            true_cand_vec: [B, D]
            isomer_cand_vecs: [B, K, D]
        Returns:
            loss: scalar Plackett-Luce cross-entropy loss
        """
        B, D = query_vec.shape
        K = isomer_cand_vecs.shape[1]

        # Normalize if metric embedding
        q = F.normalize(query_vec, p=2, dim=-1)
        t = F.normalize(true_cand_vec, p=2, dim=-1)
        neg = F.normalize(isomer_cand_vecs, p=2, dim=-1)

        # True candidate score: [B, 1]
        score_true = torch.sum(q * t, dim=-1, keepdim=True) / self.temperature

        # Isomer negative scores: [B, K]
        # q: [B, 1, D], neg: [B, K, D] -> sum over D -> [B, K]
        score_neg = torch.sum(q.unsqueeze(1) * neg, dim=-1) / self.temperature

        # All scores: [B, 1 + K] (first column is ground truth)
        all_scores = torch.cat([score_true, score_neg], dim=-1)

        # Cross entropy with target index 0
        target = torch.zeros(B, dtype=torch.long, device=query_vec.device)
        loss = F.cross_entropy(all_scores, target)

        return loss
