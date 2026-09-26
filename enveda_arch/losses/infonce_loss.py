"""
InfoNCE Contrastive Retrieval Loss (De Waele et al., ICML 2026)
Optimizes direct candidate retrieval mode to eliminate Bayes-optimal similarity-retrieval regret bounds.
Guaranteed numerically stable under AMP / fp16 autocast.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class InfoNCERetrievalLoss(nn.Module):
    """
    In-batch InfoNCE Contrastive Loss for aligning spectrum embeddings
    with true candidate molecular fingerprints vs in-batch negatives.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, spec_embeddings: torch.Tensor, cand_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spec_embeddings: [B, D] L2-normalized spectrum representations
            cand_embeddings: [B, D] L2-normalized candidate representations
        Returns:
            Scalar contrastive loss
        """
        # Ensure computation in float32
        spec_embeddings = spec_embeddings.float()
        cand_embeddings = cand_embeddings.float()

        # Normalize embeddings to unit sphere
        z_spec = F.normalize(spec_embeddings, p=2, dim=-1)
        z_cand = F.normalize(cand_embeddings, p=2, dim=-1)

        # Cosine similarity matrix [B, B]
        similarity_matrix = torch.matmul(z_spec, z_cand.t()) / self.temperature

        # Labels are diagonal (positive match is item i with item i)
        batch_size = spec_embeddings.size(0)
        labels = torch.arange(batch_size, device=spec_embeddings.device)

        # Bidirectional contrastive cross-entropy
        loss_s2c = F.cross_entropy(similarity_matrix, labels)
        loss_c2s = F.cross_entropy(similarity_matrix.t(), labels)

        return 0.5 * (loss_s2c + loss_c2s)
