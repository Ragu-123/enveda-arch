"""
Training & Validation Pipeline for enveda-arch: SpecNeuralOperatorNet on Dual Tesla T4 GPUs
Trains Multiscale Continuous Neural Operator with:
- Weighted Sparse Multi-Label Cross Entropy (25x positive bit weighting)
- Differentiable Soft Tanimoto IoU Loss
- InfoNCE Candidate Retrieval Loss (De Waele et al., ICML 2026)
- Molecular Formula Regression
- Per-Epoch Generalization Validation & CASMI MRR@25 Ranking Score
- Best Checkpoint Tracking based on Validation Loss
- Real-Time Live Plotting (Train vs Val) saved directly to /kaggle/working
"""

import os
import time
from typing import Dict, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.losses.soft_tanimoto_loss import SoftTanimotoLoss
from enveda_arch.losses.infonce_loss import InfoNCERetrievalLoss
from enveda_arch.data.dataset import EnvedaSpectraDataset, load_parquet_sample_safe
from validate import evaluate_validation

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    fp_loss_fn: nn.Module,
    tanimoto_loss_fn: nn.Module,
    infonce_loss_fn: nn.Module,
    formula_loss_fn: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    history: dict,
    epoch: int
):
    model.train()
    total_fp = 0.0
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

            # 1. Sparse multi-label loss (25x weighted for positive bits)
            loss_fp = fp_loss_fn(outputs["fingerprint_logits"], target_fp)

            # 2. Soft Tanimoto IoU loss
            loss_tani = tanimoto_loss_fn(outputs["fingerprint_logits"], target_fp)

            # 3. Decision-Theoretic InfoNCE Retrieval Loss (De Waele et al., ICML 2026)
            cand_embeds = raw_model.project_candidate_fingerprint(target_fp)
            loss_info = infonce_loss_fn(outputs["retrieval_embedding"], cand_embeds)

            # 4. Molecular formula auxiliary loss
            loss_form = formula_loss_fn(outputs["formula_preds"], target_form)

            # Unified decision-theoretic objective
            loss = loss_fp + 2.0 * loss_tani + 1.0 * loss_info + 0.1 * loss_form

        # Robust defensive check
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Step {step} produced NaN/Inf loss, skipping backward.")
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_fp += loss_fp.item()
        total_tanimoto += loss_tani.item()
        total_infonce += loss_info.item()
        total_formula += loss_form.item()
        total_loss += loss.item()

        # Record step-by-step history every 10 steps
        global_step = (epoch - 1) * len(loader) + step + 1
        if (step + 1) % 10 == 0:
            history['step'].append(global_step)
            history['total_loss'].append(loss.item())
            history['fp_loss'].append(loss_fp.item())
            history['tanimoto_loss'].append(loss_tani.item())
            history['infonce_loss'].append(loss_info.item())
            history['formula_loss'].append(loss_form.item())

        if (step + 1) % 20 == 0 or (step + 1) == len(loader):
            elapsed = time.time() - start_time
            ms_per_step = (elapsed / (step + 1)) * 1000
            print(f"  Step [{step+1:3d}/{len(loader):3d}] | Total: {loss.item():.4f} | FP: {loss_fp.item():.4f} | Tani: {loss_tani.item():.4f} | InfoNCE: {loss_info.item():.4f} | Form: {loss_form.item():.4f} | ({ms_per_step:.1f} ms/step)")

    n = len(loader)
    return total_loss / n, total_fp / n, total_tanimoto / n, total_infonce / n

def save_and_plot_convergence(
    step_history: dict,
    epoch_history: dict,
    epoch: int,
    total_epochs: int,
    is_final: bool = False,
    output_path: str = "/kaggle/working/training_convergence.png",
    csv_path: str = "/kaggle/working/training_history.csv"
):
    """
    Saves live convergence plot with both Train and Validation curves directly to /kaggle/working.
    Updates dynamically after each epoch so users can inspect progress while training.
    """
    if len(step_history['step']) == 0:
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save raw epoch history to CSV
    try:
        df_hist = pd.DataFrame(epoch_history)
        df_hist.to_csv(csv_path, index=False)
    except Exception:
        pass

    fig = plt.figure(figsize=(16, 11))
    
    steps_per_epoch = (step_history['step'][-1] // max(1, len(epoch_history['epoch']))) if len(epoch_history['epoch']) > 0 and len(step_history['step']) > 0 else 1
    epoch_steps = [e * steps_per_epoch for e in epoch_history['epoch']]

    # 1. Total Multi-Task Decision Loss (Train vs Val)
    plt.subplot(2, 2, 1)
    plt.plot(step_history['step'], step_history['total_loss'], color='#1f77b4', alpha=0.35, lw=1, label="Train (Step)")
    if len(epoch_history['epoch']) > 0:
        plt.plot(epoch_steps, epoch_history['train_loss'], 'o-', color='#1f77b4', lw=2.5, label="Train (Epoch Avg)")
        plt.plot(epoch_steps, epoch_history['val_loss'], 's--', color='#ff7f0e', lw=2.5, label="Val Loss")
    plt.title(f"Total Decision Loss (Epoch {epoch}/{total_epochs})", fontsize=12, fontweight='bold')
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 2. Morgan Fingerprint BCE Loss (Train vs Val)
    plt.subplot(2, 2, 2)
    plt.plot(step_history['step'], step_history['fp_loss'], color='#d62728', alpha=0.35, lw=1, label="Train (Step)")
    if len(epoch_history['epoch']) > 0:
        plt.plot(epoch_steps, epoch_history['train_fp'], 'o-', color='#d62728', lw=2.5, label="Train FP")
        plt.plot(epoch_steps, epoch_history['val_fp'], 's--', color='#e377c2', lw=2.5, label="Val FP")
    plt.title("Morgan Fingerprint (2048-bit) BCE Loss", fontsize=12, fontweight='bold')
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 3. Soft Tanimoto IoU Loss (Train vs Val)
    plt.subplot(2, 2, 3)
    plt.plot(step_history['step'], step_history['tanimoto_loss'], color='#2ca02c', alpha=0.35, lw=1, label="Train (Step)")
    if len(epoch_history['epoch']) > 0:
        plt.plot(epoch_steps, epoch_history['train_tanimoto'], 'o-', color='#2ca02c', lw=2.5, label="Train Tanimoto")
        plt.plot(epoch_steps, epoch_history['val_tanimoto'], 's--', color='#bcbd22', lw=2.5, label="Val Tanimoto")
    plt.title("Soft Tanimoto IoU Loss (1 - Tanimoto)", fontsize=12, fontweight='bold')
    plt.xlabel("Global Step")
    plt.ylabel("Loss")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

    # 4. InfoNCE Retrieval Loss & Validation CASMI MRR@25
    ax4 = plt.subplot(2, 2, 4)
    ax4.plot(step_history['step'], step_history['infonce_loss'], color='#9467bd', alpha=0.35, lw=1, label="Train InfoNCE (Step)")
    if len(epoch_history['epoch']) > 0:
        ax4.plot(epoch_steps, epoch_history['train_infonce'], 'o-', color='#9467bd', lw=2.5, label="Train InfoNCE")
        ax4.plot(epoch_steps, epoch_history['val_infonce'], 's--', color='#8c564b', lw=2.5, label="Val InfoNCE")
    ax4.set_title("InfoNCE Retrieval Loss & Val CASMI MRR@25", fontsize=12, fontweight='bold')
    ax4.set_xlabel("Global Step")
    ax4.set_ylabel("InfoNCE Loss", color='#9467bd')
    ax4.tick_params(axis='y', labelcolor='#9467bd')
    ax4.legend(loc='upper left')
    ax4.grid(True, alpha=0.3)

    # Secondary twin axis for CASMI MRR@25 ranking metric
    if len(epoch_history['epoch']) > 0 and 'val_mrr_25' in epoch_history:
        ax4_twin = ax4.twinx()
        ax4_twin.plot(epoch_steps, epoch_history['val_mrr_25'], 'D-', color='#17becf', lw=2.5, label="Val MRR@25")
        ax4_twin.set_ylabel("CASMI MRR@25 Score", color='#17becf')
        ax4_twin.tick_params(axis='y', labelcolor='#17becf')
        ax4_twin.legend(loc='upper right')

    status_str = "FINAL" if is_final else f"LIVE: Epoch {epoch}/{total_epochs}"
    plt.suptitle(f"SpecNeuralOperatorNet Training & Validation Convergence [{status_str}]", fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save to /kaggle/working/training_convergence.png
    plt.savefig(output_path, dpi=150, bbox_inches='tight')

    if is_final:
        plt.show()
        print(f"[OK] Final convergence plot saved to: {output_path} and displayed via logger.")
    else:
        plt.close(fig)
        print(f"[LIVE PLOT] Updated convergence plot saved to: {output_path}")

def main(epochs: int = 10, max_records: int = 25000, lr: float = 1e-3, val_ratio: float = 0.1):
    print("=" * 65)
    print(f"ENVEDA-ARCH: TRAINING SPECCRNEURALOPERATORNET ({epochs} EPOCHS, {max_records} RECORDS)")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Primary Device: {device}")
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if num_gpus > 0:
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    # 1. Safe PyArrow streaming load (Memory < 300 MB, Zero OOM)
    train_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
    print(f"\nSafely streaming up to {max_records} training records via PyArrow...")
    
    columns = [
        'ms2_mzs', 'ms2_normalized_intensities',
        'precursor_mz', 'collision_energy_ev', 'ionization_mode',
        'normalized_smiles', 'molecular_formula', 'inchikey14'
    ]
    df_raw = load_parquet_sample_safe(train_path, columns=columns, max_records=max_records)
    print(f"[OK] Safely loaded {len(df_raw)} records with zero memory spike.")

    # Grouped Split by Chemical Structure (Zero Data Leakage Guarantee)
    group_col = 'inchikey14' if 'inchikey14' in df_raw.columns and df_raw['inchikey14'].notna().any() else 'normalized_smiles'
    unique_groups = df_raw[group_col].dropna().unique()
    
    # Deterministic shuffle to ensure reproducible validation fold
    rng = np.random.RandomState(42)
    rng.shuffle(unique_groups)

    n_val_groups = max(10, int(len(unique_groups) * val_ratio))
    val_groups_set = set(unique_groups[:n_val_groups])

    val_mask = df_raw[group_col].isin(val_groups_set)
    df_val = df_raw[val_mask].reset_index(drop=True)
    df_train = df_raw[~val_mask].reset_index(drop=True)

    # Strict Zero-Leakage Verification Assertion
    train_mols = set(df_train['normalized_smiles'])
    val_mols = set(df_val['normalized_smiles'])
    mol_overlap = train_mols.intersection(val_mols)
    assert len(mol_overlap) == 0, f"CRITICAL DATA LEAKAGE: {len(mol_overlap)} structures found in both train and val!"
    
    print(f"[VERIFIED ZERO DATA LEAKAGE] Grouped strictly by '{group_col}':")
    print(f"  - Train: {len(df_train)} spectra across {len(train_mols)} unique molecules")
    print(f"  - Val:   {len(df_val)} spectra across {len(val_mols)} unique molecules")
    print(f"  - Structural Overlap: EXACTLY 0 MOLECULES (100% Leak-Free)")

    # 2. Build Datasets & DataLoaders
    batch_size = 64 if num_gpus >= 2 else 32
    print("\n--- Initializing Training Dataset ---")
    train_dataset = EnvedaSpectraDataset(df_train, max_peaks=128, n_bits=2048)
    print("\n--- Initializing Validation Dataset ---")
    val_dataset = EnvedaSpectraDataset(df_val, max_peaks=128, n_bits=2048)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == 'cuda'),
        drop_last=False
    )
    print(f"DataLoaders ready: Train={len(train_loader)} batches, Val={len(val_loader)} batches (batch_size={batch_size})")

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

    # Multi-GPU DataParallel
    if num_gpus > 1:
        print(f"Enabling DataParallel across {num_gpus} GPUs.")
        model = nn.DataParallel(model)

    # 4. Numerically Stable Weighted Losses & Optimizer
    pos_weight = torch.full((2048,), 25.0, device=device)
    fp_loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    tanimoto_loss_fn = SoftTanimotoLoss()
    infonce_loss_fn = InfoNCERetrievalLoss(temperature=0.07)
    formula_loss_fn = nn.SmoothL1Loss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    step_history = {
        'step': [],
        'total_loss': [],
        'fp_loss': [],
        'tanimoto_loss': [],
        'infonce_loss': [],
        'formula_loss': []
    }
    
    epoch_history = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'train_fp': [],
        'val_fp': [],
        'train_tanimoto': [],
        'val_tanimoto': [],
        'train_infonce': [],
        'val_infonce': [],
        'val_mrr_25': [],
        'val_top1_acc': [],
        'val_top25_acc': []
    }

    checkpoint_dir = "/kaggle/working/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_loss = float('inf')

    # 5. Training & Per-Epoch Validation Loop
    print("\n" + "=" * 65)
    print(f"BEGINNING TRAINING & PER-EPOCH VALIDATION ({epochs} EPOCHS)")
    print("=" * 65)

    raw_model = model.module if hasattr(model, 'module') else model

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        print(f"\n--- Epoch {epoch}/{epochs} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        
        # 1. Training epoch
        train_loss, fp_l, tani, info = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            fp_loss_fn=fp_loss_fn,
            tanimoto_loss_fn=tanimoto_loss_fn,
            infonce_loss_fn=infonce_loss_fn,
            formula_loss_fn=formula_loss_fn,
            scaler=scaler,
            device=device,
            history=step_history,
            epoch=epoch
        )
        scheduler.step()

        # 2. Per-Epoch Validation Evaluation
        val_metrics = evaluate_validation(
            model=model,
            loader=val_loader,
            fp_loss_fn=fp_loss_fn,
            tanimoto_loss_fn=tanimoto_loss_fn,
            infonce_loss_fn=infonce_loss_fn,
            formula_loss_fn=formula_loss_fn,
            device=device
        )

        ep_time = time.time() - t0
        print(f"Epoch {epoch:2d}/{epochs:2d} ({ep_time:.1f}s) | "
              f"Train Loss: {train_loss:.4f} (InfoNCE: {info:.4f}) | "
              f"Val Loss: {val_metrics['val_loss']:.4f} (InfoNCE: {val_metrics['val_infonce_loss']:.4f}) | "
              f"Val MRR@25: {val_metrics['mrr_25']:.4f} | Top-25: {val_metrics['top25_acc']*100:.1f}%")

        # Record epoch metrics
        epoch_history['epoch'].append(epoch)
        epoch_history['train_loss'].append(train_loss)
        epoch_history['val_loss'].append(val_metrics['val_loss'])
        epoch_history['train_fp'].append(fp_l)
        epoch_history['val_fp'].append(val_metrics['val_fp_loss'])
        epoch_history['train_tanimoto'].append(tani)
        epoch_history['val_tanimoto'].append(val_metrics['val_tanimoto_loss'])
        epoch_history['train_infonce'].append(info)
        epoch_history['val_infonce'].append(val_metrics['val_infonce_loss'])
        epoch_history['val_mrr_25'].append(val_metrics['mrr_25'])
        epoch_history['val_top1_acc'].append(val_metrics['top1_acc'])
        epoch_history['val_top25_acc'].append(val_metrics['top25_acc'])

        # Save best model checkpoint
        if val_metrics['val_loss'] < best_val_loss:
            best_val_loss = val_metrics['val_loss']
            ckpt_best = os.path.join(checkpoint_dir, "spec_neural_operator_best.pt")
            torch.save(raw_model.state_dict(), ckpt_best)
            print(f"  --> [NEW BEST MODEL] Saved to: {ckpt_best} (Val Loss: {best_val_loss:.4f})")

        # Always save latest checkpoint
        ckpt_latest = os.path.join(checkpoint_dir, "spec_neural_operator.pt")
        torch.save(raw_model.state_dict(), ckpt_latest)

        # Update live convergence plot & CSV in /kaggle/working
        is_final = (epoch == epochs)
        save_and_plot_convergence(
            step_history=step_history,
            epoch_history=epoch_history,
            epoch=epoch,
            total_epochs=epochs,
            is_final=is_final,
            output_path="/kaggle/working/training_convergence.png",
            csv_path="/kaggle/working/training_history.csv"
        )

    print(f"\n[COMPLETE] Training finished. Best Val Loss: {best_val_loss:.4f}")
    print(f"  - Checkpoints: /kaggle/working/checkpoints/spec_neural_operator_best.pt")
    print(f"  - Convergence Plot: /kaggle/working/training_convergence.png")
    print(f"  - History CSV: /kaggle/working/training_history.csv")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train SpecNeuralOperatorNet on MS/MS Spectra")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs (default: 10)")
    parser.add_argument("--max_records", type=int, default=25000, help="Number of spectra to load (default: 25000)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Peak learning rate (default: 1e-3)")
    parser.add_argument("--val_ratio", type=float, default=0.1, help="Validation ratio (default: 0.1)")
    args = parser.parse_args()
    main(epochs=args.epochs, max_records=args.max_records, lr=args.lr, val_ratio=args.val_ratio)
