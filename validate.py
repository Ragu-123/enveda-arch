"""
Validation & Ranking Evaluation Pipeline for SpecNeuralOperatorNet
Evaluates multi-task loss components and CASMI MRR@25 / Top-k retrieval metrics.
Can be invoked per-epoch by train.py or executed standalone as a CLI tool.
"""

import os
import time
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.losses.asymmetric_loss import AsymmetricLoss
from enveda_arch.losses.soft_tanimoto_loss import SoftTanimotoLoss
from enveda_arch.losses.infonce_loss import InfoNCERetrievalLoss
from enveda_arch.data.dataset import EnvedaSpectraDataset, load_parquet_sample_safe

@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    fp_loss_fn: nn.Module,
    tanimoto_loss_fn: nn.Module,
    infonce_loss_fn: nn.Module,
    formula_loss_fn: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """
    Evaluates model across validation loader and computes both losses and ranking metrics (MRR@25, Top-1, Top-25).
    """
    model.eval()
    raw_model = model.module if hasattr(model, 'module') else model

    total_loss = 0.0
    total_fp = 0.0
    total_tani = 0.0
    total_infonce = 0.0
    total_formula = 0.0
    
    top1_hits = 0
    top5_hits = 0
    top25_hits = 0
    reciprocal_ranks = []
    total_samples = 0

    from tqdm import tqdm
    pbar = tqdm(loader, desc="Validating (MRR@25)", total=len(loader), dynamic_ncols=True, leave=False)

    for batch in pbar:
        mzs = batch["mzs"].to(device)
        intensities = batch["intensities"].to(device)
        precursor_mz = batch["precursor_mz"].to(device)
        collision_energy = batch["collision_energy"].to(device)
        mode = batch["mode"].to(device)
        mask = batch["mask"].to(device)
        target_fp = batch["target_fingerprint"].to(device)
        target_form = batch["target_formula"].to(device)
        b_size = mzs.size(0)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(
                mzs=mzs,
                intensities=intensities,
                precursor_mz=precursor_mz,
                collision_energy=collision_energy,
                mode=mode,
                mask=mask
            )

            # 1. Multi-task loss terms
            loss_fp = fp_loss_fn(outputs["fingerprint_logits"], target_fp)
            loss_tani = tanimoto_loss_fn(outputs["fingerprint_logits"], target_fp)
            cand_embeds = raw_model.project_candidate_fingerprint(target_fp)
            loss_info = infonce_loss_fn(outputs["retrieval_embedding"], cand_embeds)
            loss_form = formula_loss_fn(outputs["formula_preds"], target_form)

            batch_loss = 5.0 * loss_fp + 1.0 * loss_info + 0.1 * loss_form

        total_loss += batch_loss.item()
        total_fp += loss_fp.item()
        total_tani += loss_tani.item()
        total_infonce += loss_info.item()
        total_formula += loss_form.item()

        # 2. Ranking Evaluation within batch
        # Cosine similarity matrix: [B, B]
        spec_embeds = outputs["retrieval_embedding"]
        sim_matrix = torch.matmul(spec_embeds, cand_embeds.T) # [B, B]
        
        # Rank of ground-truth candidate (diagonal element i, i)
        ranks = torch.argsort(sim_matrix, dim=-1, descending=True) # [B, B]
        target_indices = torch.arange(b_size, device=device).unsqueeze(1) # [B, 1]
        matches = (ranks == target_indices).nonzero(as_tuple=True)[1].cpu().numpy() # [B] rank positions (0-indexed)

        for rank_pos in matches:
            rank_1_indexed = rank_pos + 1
            if rank_1_indexed == 1:
                top1_hits += 1
            if rank_1_indexed <= 5:
                top5_hits += 1
            if rank_1_indexed <= 25:
                top25_hits += 1
                reciprocal_ranks.append(1.0 / rank_1_indexed)
            else:
                reciprocal_ranks.append(0.0)

        total_samples += b_size

    n_batches = max(1, len(loader))
    mrr_25 = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0

    return {
        "val_loss": total_loss / n_batches,
        "val_fp_loss": total_fp / n_batches,
        "val_tanimoto_loss": total_tani / n_batches,
        "val_infonce_loss": total_infonce / n_batches,
        "val_formula_loss": total_formula / n_batches,
        "top1_acc": top1_hits / max(1, total_samples),
        "top5_acc": top5_hits / max(1, total_samples),
        "top25_acc": top25_hits / max(1, total_samples),
        "mrr_25": mrr_25
    }

def main(
    checkpoint_path: str = "/kaggle/working/checkpoints/spec_neural_operator.pt",
    val_parquet_path: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet",
    max_records: int = 3000,
    batch_size: int = 64
):
    print("=" * 65)
    print("ENVEDA-ARCH: STANDALONE VALIDATION EVALUATION")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load validation slice
    print(f"\nLoading {max_records} validation records via PyArrow...")
    columns = [
        'ms2_mzs', 'ms2_normalized_intensities',
        'precursor_mz', 'collision_energy_ev', 'ionization_mode',
        'normalized_smiles', 'molecular_formula'
    ]
    df_val = load_parquet_sample_safe(val_parquet_path, columns=columns, max_records=max_records)
    val_dataset = EnvedaSpectraDataset(df_val, max_peaks=128, n_bits=2048)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    print(f"[OK] Validation loader ready with {len(val_dataset)} records.")

    # Load Model
    model = SpecNeuralOperatorNet(
        hidden_dim=256,
        retrieval_dim=2048,
        fingerprint_dim=2048,
        formula_dim=10,
        num_operator_layers=2,
        sigmas=(0.02, 0.5, 5.0, 28.0),
        num_heads=4,
        fourier_dim=128
    )
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    print("[OK] Checkpoint loaded successfully.")

    # Loss functions
    fp_loss_fn = AsymmetricLoss(gamma_neg=2.0, gamma_pos=0.0, clip=0.05)
    tanimoto_loss_fn = SoftTanimotoLoss()
    infonce_loss_fn = InfoNCERetrievalLoss(temperature=0.15)
    formula_loss_fn = nn.SmoothL1Loss()

    t0 = time.time()
    metrics = evaluate_validation(
        model=model,
        loader=val_loader,
        fp_loss_fn=fp_loss_fn,
        tanimoto_loss_fn=tanimoto_loss_fn,
        infonce_loss_fn=infonce_loss_fn,
        formula_loss_fn=formula_loss_fn,
        device=device
    )
    elapsed = time.time() - t0

    print("\n" + "=" * 65)
    print("VALIDATION METRICS SUMMARY")
    print("=" * 65)
    print(f"Evaluation Time: {elapsed:.2f}s ({len(val_dataset) / elapsed:.1f} spectra/sec)")
    print(f"Total Loss:          {metrics['val_loss']:.4f}")
    print(f"InfoNCE Loss:        {metrics['val_infonce_loss']:.4f}")
    print(f"Soft Tanimoto Loss:  {metrics['val_tanimoto_loss']:.4f}")
    print(f"Morgan BCE Loss:     {metrics['val_fp_loss']:.4f}")
    print(f"Formula Loss:        {metrics['val_formula_loss']:.4f}")
    print("-" * 65)
    print(f"Top-1 Accuracy:      {metrics['top1_acc'] * 100:.2f}%")
    print(f"Top-5 Accuracy:      {metrics['top5_acc'] * 100:.2f}%")
    print(f"Top-25 Accuracy:     {metrics['top25_acc'] * 100:.2f}%")
    print(f"CASMI MRR@25 Score:  {metrics['mrr_25']:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate SpecNeuralOperatorNet")
    parser.add_argument("--checkpoint", type=str, default="/kaggle/working/checkpoints/spec_neural_operator.pt")
    parser.add_argument("--max_records", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint, max_records=args.max_records, batch_size=args.batch_size)
