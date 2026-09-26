"""
Training Pipeline for enveda-arch: SpecNeuralOperatorNet on Dual Tesla T4 GPUs
Trains Multiscale Continuous Neural Operator with:
- Asymmetric Multi-Label Loss (ASL)
- Soft Tanimoto IoU Loss
- InfoNCE Candidate Retrieval Loss (eliminating Bayes retrieval regret)
- Molecular Formula Regression
- Zero Host-RAM OOM Streaming via PyArrow
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.losses.asymmetric_loss import AsymmetricLoss
from enveda_arch.losses.soft_tanimoto_loss import SoftTanimotoLoss
from enveda_arch.losses.infonce_loss import InfoNCERetrievalLoss
from enveda_arch.data.dataset import EnvedaSpectraDataset, load_parquet_sample_safe

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    asl_loss_fn: nn.Module,
    tanimoto_loss_fn: nn.Module,
    infonce_loss_fn: nn.Module,
    formula_loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device
):
    model.train()
    total_asl = 0.0
    total_tanimoto = 0.0
    total_infonce = 0.0
    total_formula = 0.0
    total_loss = 0.0
    start_time = time.time()

    raw_model = model.module if hasattr(model, 'module') else model

    for step, batch in enumerate(loader):
        mzs = batch["mzs"].to(device)
        intensities = batch["intensities"].to(device)
        precursor_mz = batch["precursor_mz"].to(device)
        collision_energy = batch["collision_energy"].to(device)
        mode = batch["mode"].to(device)
        mask = batch["mask"].to(device)
        
        target_fp = batch["target_fingerprint"].to(device)
        target_form = batch["target_formula"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast('cuda', enabled=device.type == 'cuda'):
            outputs = model(
                mzs=mzs,
                intensities=intensities,
                precursor_mz=precursor_mz,
                collision_energy=collision_energy,
                mode=mode,
                mask=mask
            )

            # 1. Asymmetric multi-label loss
            loss_asl = asl_loss_fn(outputs["fingerprint_logits"], target_fp)

            # 2. Soft Tanimoto IoU loss
            loss_tani = tanimoto_loss_fn(outputs["fingerprint_logits"], target_fp)

            # 3. Decision-Theoretic InfoNCE Retrieval Loss (De Waele et al., ICML 2026)
            cand_embeds = raw_model.project_candidate_fingerprint(target_fp)
            loss_info = infonce_loss_fn(outputs["retrieval_embedding"], cand_embeds)

            # 4. Molecular formula auxiliary loss
            loss_form = formula_loss_fn(outputs["formula_preds"], target_form)

            # Unified decision-theoretic objective
            loss = 5.0 * loss_asl + 2.0 * loss_tani + 1.0 * loss_info + 0.1 * loss_form

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_asl += loss_asl.item()
        total_tanimoto += loss_tani.item()
        total_infonce += loss_info.item()
        total_formula += loss_form.item()
        total_loss += loss.item()

        if (step + 1) % 20 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - start_time
            ms_per_step = (elapsed / (step + 1)) * 1000
            print(f"  Step [{step+1:3d}/{len(loader):3d}] | Total: {loss.item():.4f} | ASL: {loss_asl.item():.4f} | Tani: {loss_tani.item():.4f} | InfoNCE: {loss_info.item():.4f} | Form: {loss_form.item():.4f} | ({ms_per_step:.1f} ms/step)")

    n = len(loader)
    return total_loss / n, total_asl / n, total_tanimoto / n, total_infonce / n

def main():
    print("=" * 65)
    print("ENVEDA-ARCH: TRAINING SPECCRNEURALOPERATORNET")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Primary Device: {device}")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 1. Safe PyArrow streaming load (Memory < 300 MB, Zero OOM)
    train_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
    print("\nSafely streaming training records via PyArrow...")
    
    columns = [
        'ms2_mzs', 'ms2_normalized_intensities',
        'precursor_mz', 'collision_energy_ev', 'ionization_mode',
        'normalized_smiles', 'molecular_formula'
    ]
    df_train = load_parquet_sample_safe(train_path, columns=columns, max_records=25000)
    print(f"[OK] Safely loaded {len(df_train)} training records with zero memory spike.")

    # 2. Build Dataset & DataLoader
    # Batch size scaled for Dual Tesla T4 GPUs
    batch_size = 64 if num_gpus >= 2 else 32
    dataset = EnvedaSpectraDataset(df_train, max_peaks=128, n_bits=2048)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    print(f"DataLoader initialized: {len(loader)} batches of size {batch_size}")

    # 3. Instantiate Novel Architecture
    model = SpecNeuralOperatorNet(
        hidden_dim=256,
        retrieval_dim=256,
        fingerprint_dim=2048,
        formula_dim=10,
        num_operator_layers=2,
        sigmas=(0.02, 0.5, 5.0, 28.0),
        num_heads=4,
        fourier_dim=128
    )
    
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Initialized SpecNeuralOperatorNet: {total_params:,} trainable parameters.")

    model = model.to(device)

    # Multi-GPU support
    if num_gpus > 1:
        print(f"Enabling DataParallel across {num_gpus} GPUs.")
        model = nn.DataParallel(model)

    # 4. Losses & Optimizer
    asl_loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    tanimoto_loss_fn = SoftTanimotoLoss()
    infonce_loss_fn = InfoNCERetrievalLoss(temperature=0.07)
    formula_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    # 5. Training Loop
    epochs = 3
    print("\n" + "=" * 65)
    print("BEGINNING TRAINING EPOCHS")
    print("=" * 65)

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        train_loss, asl, tani, info = train_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            asl_loss_fn=asl_loss_fn,
            tanimoto_loss_fn=tanimoto_loss_fn,
            infonce_loss_fn=infonce_loss_fn,
            formula_loss_fn=formula_loss_fn,
            scaler=scaler,
            device=device
        )
        scheduler.step()
        ep_time = time.time() - t0
        print(f"Epoch {epoch} Complete in {ep_time:.1f}s | Avg Loss: {train_loss:.4f} (ASL: {asl:.4f}, Tani: {tani:.4f}, InfoNCE: {info:.4f})")

    # 6. Save Model Checkpoint
    checkpoint_dir = "/kaggle/working/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoint_dir, "spec_neural_operator.pt")
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save(raw_model.state_dict(), ckpt_path)
    print(f"\n[SUCCESS] Model checkpoint saved to: {ckpt_path}")

if __name__ == "__main__":
    main()
