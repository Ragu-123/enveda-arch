"""
Validation & Ranking Evaluation Pipeline for SpecNeuralOperatorNet
Evaluates:
1. Multi-task loss components (InfoNCE, Soft Tanimoto, ASL BCE, Formula)
2. In-Batch Open-Search MRR@25
3. Realistic CASMI Regime B Mass-Windowed Candidate Ranking MRR@25 (+-8.5 ppm candidate pool)
"""

import os
import time
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.losses.asymmetric_loss import AsymmetricLoss
from enveda_arch.losses.soft_tanimoto_loss import SoftTanimotoLoss
from enveda_arch.losses.infonce_loss import InfoNCERetrievalLoss
from enveda_arch.data.dataset import EnvedaSpectraDataset, load_parquet_sample_safe
from enveda_arch.retrieval import (
    build_candidate_index_from_train,
    retrieve_top_k_candidates,
    get_neutral_mass_from_adduct,
    compute_inchikey14,
    PROTON_MASS
)

@torch.no_grad()
def evaluate_validation(
    model: nn.Module,
    loader: DataLoader,
    fp_loss_fn: nn.Module,
    tanimoto_loss_fn: nn.Module,
    infonce_loss_fn: nn.Module,
    formula_loss_fn: nn.Module,
    device: torch.device,
    fwd_loss_fn: Optional[nn.Module] = None,
    cand_index: Optional[Tuple[np.ndarray, np.ndarray, List[str], List[str]]] = None,
    eval_regime_b_samples: int = 150
) -> Dict[str, float]:
    """
    Evaluates model across validation loader and computes both in-batch loss metrics
    AND realistic CASMI Mass-Windowed MRR@25 (Regime B benchmark: distinguishing the true structure
    among mass-matched isomers within +-8.5 ppm).
    """
    model.eval()
    raw_model = model.module if hasattr(model, 'module') else model

    total_loss = 0.0
    total_fp = 0.0
    total_tani = 0.0
    total_infonce = 0.0
    total_formula = 0.0
    total_fwd = 0.0
    
    inbatch_top1 = 0
    inbatch_top5 = 0
    inbatch_top25 = 0
    inbatch_mrr_list = []
    total_samples = 0

    # Regime B mass-window tracking
    regime_b_top1 = 0
    regime_b_top5 = 0
    regime_b_top25 = 0
    regime_b_mrr_list = []
    regime_b_count = 0

    cand_masses, cand_fps, cand_smiles, cand_ik14 = cand_index if cand_index is not None else (None, None, None, None)

    from tqdm import tqdm
    from enveda_arch.losses import build_spectral_density_target
    pbar = tqdm(loader, desc="Validating (Loss & Ranking)", total=len(loader), dynamic_ncols=True, leave=False)

    for batch in pbar:
        mzs = batch["mzs"].to(device)
        intensities = batch["intensities"].to(device)
        precursor_mz = batch["precursor_mz"].to(device)
        collision_energy = batch["collision_energy"].to(device)
        mode = batch["mode"].to(device)
        mask = batch["mask"].to(device)
        adduct_ix = batch.get("adduct_ix", None)
        instr_ix = batch.get("instr_ix", None)
        if adduct_ix is not None:
            adduct_ix = adduct_ix.to(device)
        if instr_ix is not None:
            instr_ix = instr_ix.to(device)
        target_fp = batch["target_fingerprint"].to(device)
        target_form = batch["target_formula"].to(device)
        target_smiles = batch.get("smiles", None)
        b_size = mzs.size(0)

        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            outputs = model(
                mzs=mzs,
                intensities=intensities,
                precursor_mz=precursor_mz,
                collision_energy=collision_energy,
                mode=mode,
                mask=mask,
                adduct_ix=adduct_ix,
                instr_ix=instr_ix
            )

            # 1. Multi-task loss terms
            loss_fp = fp_loss_fn(outputs["fingerprint_logits"], target_fp)
            loss_tani = tanimoto_loss_fn(outputs["fingerprint_logits"], target_fp)
            cand_embeds = raw_model.project_candidate_fingerprint(target_fp)
            loss_info = infonce_loss_fn(outputs["retrieval_embedding"], cand_embeds)
            loss_form = formula_loss_fn(outputs["formula_preds"], target_form)

            if fwd_loss_fn is not None:
                target_density = build_spectral_density_target(mzs, intensities, mask=mask, num_bins=512)
                fwd_density = raw_model.predict_forward_spectrum(target_fp, precursor_mz, collision_energy, mode)
                loss_fwd = fwd_loss_fn(fwd_density, target_density)
            else:
                loss_fwd = torch.tensor(0.0, device=device)

            batch_loss = 1.0 * loss_fp + 2.0 * loss_tani + 1.0 * loss_info + 0.1 * loss_form + 0.5 * loss_fwd

        total_loss += batch_loss.item()
        total_fp += loss_fp.item()
        total_tani += loss_tani.item()
        total_infonce += loss_info.item()
        total_formula += loss_form.item()
        total_fwd += loss_fwd.item()

        # 2. In-Batch Open-Search Ranking Evaluation
        spec_embeds = outputs["retrieval_embedding"]
        sim_matrix = torch.matmul(spec_embeds, cand_embeds.T)  # [B, B]
        ranks = torch.argsort(sim_matrix, dim=-1, descending=True)
        target_indices = torch.arange(b_size, device=device).unsqueeze(1)
        matches = (ranks == target_indices).nonzero(as_tuple=True)[1].cpu().numpy()

        for rank_pos in matches:
            r = rank_pos + 1
            if r == 1:
                inbatch_top1 += 1
            if r <= 5:
                inbatch_top5 += 1
            if r <= 25:
                inbatch_top25 += 1
                inbatch_mrr_list.append(1.0 / r)
            else:
                inbatch_mrr_list.append(0.0)

        total_samples += b_size

        # 3. Realistic Regime B Mass-Window Candidate Ranking Evaluation
        if cand_masses is not None and target_smiles is not None and regime_b_count < eval_regime_b_samples:
            for b_idx in range(b_size):
                if regime_b_count >= eval_regime_b_samples:
                    break

                gt_smiles = target_smiles[b_idx]
                gt_ik14 = compute_inchikey14(gt_smiles)
                m_val = float(precursor_mz[b_idx].item())
                mode_v = float(mode[b_idx].item())
                neutral_mass = m_val - PROTON_MASS if mode_v > 0.5 else m_val + PROTON_MASS

                # Window +-8.5 ppm (optimal per community findings)
                tol_da = neutral_mass * (8.5 * 1e-6)
                idx_l = np.searchsorted(cand_masses, neutral_mass - tol_da)
                idx_r = np.searchsorted(cand_masses, neutral_mass + tol_da)
                
                # Ensure minimum 15 candidates for a realistic isomer test
                if (idx_r - idx_l) < 15:
                    tol_da = neutral_mass * (25.0 * 1e-6)
                    idx_l = np.searchsorted(cand_masses, neutral_mass - tol_da)
                    idx_r = np.searchsorted(cand_masses, neutral_mass + tol_da)

                sub_fps = list(cand_fps[idx_l:idx_r])
                sub_smiles = list(np.array(cand_smiles)[idx_l:idx_r])
                sub_ik14 = list(np.array(cand_ik14)[idx_l:idx_r]) if cand_ik14 is not None else [compute_inchikey14(s) for s in sub_smiles]

                # Inject ground truth if not in slice
                if gt_ik14 not in sub_ik14:
                    gt_fp = target_fp[b_idx].cpu().numpy()
                    sub_fps.append(gt_fp)
                    sub_smiles.append(gt_smiles)
                    sub_ik14.append(gt_ik14)

                sub_fps_t = torch.tensor(np.array(sub_fps), dtype=torch.float32, device=device)
                q_embed = spec_embeds[b_idx:b_idx+1].float()
                q_logits = outputs["fingerprint_logits"][b_idx:b_idx+1].float()

                # Ranking using unified score
                cand_sub_embeds = F.normalize(sub_fps_t, p=2, dim=-1)
                sim_ret = torch.matmul(cand_sub_embeds, q_embed.squeeze(0)).cpu().numpy()

                z_query = q_logits.view(-1).float()
                bayes_scores = torch.matmul(sub_fps_t, z_query).cpu().numpy()
                b_min, b_max = bayes_scores.min(), bayes_scores.max()
                bayes_norm = (bayes_scores - b_min) / (b_max - b_min + 1e-6)

                spec_probs = torch.sigmoid(z_query)
                intersection = torch.sum(sub_fps_t * spec_probs, dim=-1)
                union = torch.sum(sub_fps_t + spec_probs - (sub_fps_t * spec_probs), dim=-1)
                sim_tani = (intersection / (union + 1e-6)).cpu().numpy()

                score = 0.40 * sim_ret + 0.35 * bayes_norm + 0.25 * sim_tani
                ranked_cand_idx = np.argsort(score)[::-1]

                # Deduplicate by InChIKey14 and find GT rank
                seen_k = set()
                gt_rank = None
                curr_rank = 1

                for c_i in ranked_cand_idx:
                    k14 = sub_ik14[c_i]
                    if k14 in seen_k:
                        continue
                    seen_k.add(k14)
                    if k14 == gt_ik14:
                        gt_rank = curr_rank
                        break
                    curr_rank += 1
                    if curr_rank > 25:
                        break

                if gt_rank is not None and gt_rank <= 25:
                    if gt_rank == 1:
                        regime_b_top1 += 1
                    if gt_rank <= 5:
                        regime_b_top5 += 1
                    regime_b_top25 += 1
                    regime_b_mrr_list.append(1.0 / gt_rank)
                else:
                    regime_b_mrr_list.append(0.0)

                regime_b_count += 1

    n_batches = max(1, len(loader))
    inbatch_mrr = float(np.mean(inbatch_mrr_list)) if inbatch_mrr_list else 0.0
    regime_b_mrr = float(np.mean(regime_b_mrr_list)) if regime_b_mrr_list else 0.0

    return {
        "val_loss": total_loss / n_batches,
        "val_fp_loss": total_fp / n_batches,
        "val_tanimoto_loss": total_tani / n_batches,
        "val_infonce_loss": total_infonce / n_batches,
        "val_formula_loss": total_formula / n_batches,
        "inbatch_top1": inbatch_top1 / max(1, total_samples),
        "inbatch_top5": inbatch_top5 / max(1, total_samples),
        "inbatch_top25": inbatch_top25 / max(1, total_samples),
        "inbatch_mrr25": inbatch_mrr,
        "regime_b_samples": regime_b_count,
        "regime_b_top1": regime_b_top1 / max(1, regime_b_count),
        "regime_b_top5": regime_b_top5 / max(1, regime_b_count),
        "regime_b_top25": regime_b_top25 / max(1, regime_b_count),
        "regime_b_mrr25": regime_b_mrr,
        "mrr_25": regime_b_mrr if regime_b_count > 0 else inbatch_mrr,
        "top1_acc": regime_b_top1 / max(1, regime_b_count) if regime_b_count > 0 else inbatch_top1 / max(1, total_samples),
        "top25_acc": regime_b_top25 / max(1, regime_b_count) if regime_b_count > 0 else inbatch_top25 / max(1, total_samples)
    }

def main(
    checkpoint_path: str = "/kaggle/working/checkpoints/spec_neural_operator.pt",
    val_parquet_path: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet",
    max_records: int = 3000,
    batch_size: int = 64
):
    print("=" * 65)
    print("ENVEDA-ARCH: DUAL REGIME VALIDATION EVALUATION")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # Load candidate index for Regime B evaluation
    cand_index = None
    cand_index_path = "/kaggle/working/candidate_index_merged.npz"
    if not os.path.exists(cand_index_path):
        cand_index_path = "/kaggle/working/candidate_index.npz"
    if os.path.exists(cand_index_path):
        print(f"Loading candidate index from {cand_index_path}...")
        data = np.load(cand_index_path, allow_pickle=True)
        cand_index = (
            data["masses"],
            data["fps"],
            list(data["smiles"]),
            list(data["ik14"]) if "ik14" in data.files else None
        )
        print(f"[OK] Loaded {len(data['masses'])} candidate structures.")

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
        device=device,
        cand_index=cand_index,
        eval_regime_b_samples=150
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
    print("IN-BATCH OPEN SEARCH METRICS (UNCONSTRAINED DECOYS):")
    print(f"  Top-1 Accuracy:    {metrics['inbatch_top1'] * 100:.2f}%")
    print(f"  Top-5 Accuracy:    {metrics['inbatch_top5'] * 100:.2f}%")
    print(f"  Top-25 Accuracy:   {metrics['inbatch_top25'] * 100:.2f}%")
    print(f"  In-Batch MRR@25:   {metrics['inbatch_mrr25']:.4f}")
    print("-" * 65)
    if metrics["regime_b_samples"] > 0:
        print(f"REGIME B MASS-WINDOWED METRICS (+-8.5 ppm, {metrics['regime_b_samples']} molecules):")
        print(f"  Top-1 Accuracy:    {metrics['regime_b_top1'] * 100:.2f}%")
        print(f"  Top-5 Accuracy:    {metrics['regime_b_top5'] * 100:.2f}%")
        print(f"  Top-25 Accuracy:   {metrics['regime_b_top25'] * 100:.2f}%")
        print(f"  CASMI MRR@25:      {metrics['regime_b_mrr25']:.4f}")
    print("=" * 65)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Validate SpecNeuralOperatorNet")
    parser.add_argument("--checkpoint", type=str, default="/kaggle/working/checkpoints/spec_neural_operator.pt")
    parser.add_argument("--max_records", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()
    main(checkpoint_path=args.checkpoint, max_records=args.max_records, batch_size=args.batch_size)
