"""
Training Pipeline for enveda-arch: SpecContinuousNet on Kaggle Dual GPUs
Trains continuous MS/MS spectrum-to-fingerprint network with Asymmetric Loss (ASL) & Soft Tanimoto.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np

from enveda_arch.models.spec_net import SpecContinuousNet
from enveda_arch.losses.asymmetric_loss import AsymmetricLoss
from enveda_arch.losses.soft_tanimoto_loss import SoftTanimotoLoss
from enveda_arch.data.dataset import EnvedaSpectraDataset

def train_epoch(model, loader, optimizer, asl_loss_fn, tanimoto_loss_fn, formula_loss_fn, scaler, device):
    model.train()
    total_asl = 0.0
    total_tanimoto = 0.0
    total_formula = 0.0
    total_loss = 0.0
    start_time = time.time()

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

            loss_asl = asl_loss_fn(outputs["fingerprint_logits"], target_fp)
            loss_tani = tanimoto_loss_fn(outputs["fingerprint_logits"], target_fp)
            loss_form = formula_loss_fn(outputs["formula_preds"], target_form)

            # Combined multi-task loss
            loss = loss_asl + 2.0 * loss_tani + 0.1 * loss_form

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_asl += loss_asl.item()
        total_tanimoto += loss_tani.item()
        total_formula += loss_form.item()
        total_loss += loss.item()

        if (step + 1) % 25 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - start_time
            ms_per_step = (elapsed / (step + 1)) * 1000
            print(f"  Step [{step+1:3d}/{len(loader):3d}] | Total: {loss.item():.4f} | ASL: {loss_asl.item():.4f} | Tanimoto: {loss_tani.item():.4f} | Formula: {loss_form.item():.4f} | ({ms_per_step:.1f} ms/step)")

    n = len(loader)
    return total_loss / n, total_asl / n, total_tanimoto / n

def main():
    print("=" * 65)
    print("ENVEDA-ARCH: TRAINING SPECCONTINUOUSNET")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # Load training sample (using in-house timsTOF spectra from enveda-180 and enveda-np-examples)
    train_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
    print("\nLoading training batch from train.parquet...")
    
    # Load first 20,000 spectra for high-speed convergence testing
    df_chunk = pd.read_parquet(
        train_path,
        columns=[
            'ingest_lib', 'ms2_mzs', 'ms2_normalized_intensities',
            'precursor_mz', 'collision_energy_ev', 'ionization_mode',
            'normalized_smiles', 'molecular_formula'
        ]
    ).head(20000)
    print(f"Loaded {len(df_chunk)} training records.")

    dataset = EnvedaSpectraDataset(df_chunk, max_peaks=128, n_bits=2048)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)

    # Initialize SpecContinuousNet
    model = SpecContinuousNet(d_model=256, num_layers=4, cond_dim=3, fingerprint_dim=2048)
    if torch.cuda.device_count() > 1:
        print(f"Wrapping model with DataParallel across {torch.cuda.device_count()} GPUs...")
        model = nn.DataParallel(model)
    model.to(device)

    # Loss Functions
    asl_loss_fn = AsymmetricLoss(gamma_neg=4.0, gamma_pos=1.0, clip=0.05)
    tanimoto_loss_fn = SoftTanimotoLoss()
    formula_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda')

    epochs = 3
    print(f"\nStarting {epochs} epochs of high-throughput training...")
    for epoch in range(1, epochs + 1):
        print(f"\n--- Epoch {epoch}/{epochs} ---")
        t0 = time.time()
        loss, asl, tani = train_epoch(
            model, loader, optimizer, asl_loss_fn, tanimoto_loss_fn, formula_loss_fn, scaler, device
        )
        print(f"Epoch {epoch} Complete | Avg Loss: {loss:.4f} | Avg ASL: {asl:.4f} | Soft Tanimoto Loss: {tani:.4f} | Duration: {time.time()-t0:.1f}s")

    # Save checkpoint
    save_path = "/kaggle/working/spec_continuous_net_checkpoint.pt"
    raw_model = model.module if hasattr(model, 'module') else model
    torch.save(raw_model.state_dict(), save_path)
    print(f"\n[OK] Model weights successfully saved to {save_path}")

if __name__ == "__main__":
    main()
