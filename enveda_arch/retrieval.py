"""
Fast Binary-Search Candidate Retrieval & Metric Ranking Engine
Given trained SpecNeuralOperatorNet, retrieves candidate molecules from pre-sorted neutral mass index
and ranks top 25 candidates using hybrid InfoNCE metric embedding + Morgan fingerprint similarity.
"""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.data.dataset import smiles_to_morgan_fingerprint

PROTON_MASS = 1.007276

def build_candidate_index_from_train(
    train_parquet_path: str,
    max_records: int = 150000,
    save_path: str = "/kaggle/working/candidate_index.npz"
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Builds a pre-sorted candidate database of (neutral_masses, fingerprints, smiles).
    Uses PyArrow row-group streaming to prevent RAM exhaustion.
    """
    if os.path.exists(save_path):
        print(f"Loading cached candidate index from {save_path}...")
        data = np.load(save_path, allow_pickle=True)
        fps = data["fps"]
        if fps.sum() > 0:
            print(f"[OK] Loaded valid candidate index ({len(data['masses'])} candidates, mean active bits: {fps.sum(axis=1).mean():.1f}).")
            return data["masses"], fps, list(data["smiles"])
        else:
            print(f"[WARNING] Cached candidate index at {save_path} has all-zero fingerprints! Rebuilding...")

    print(f"Building candidate index from {train_parquet_path}...")
    pf = pq.ParquetFile(train_parquet_path)
    
    unique_smiles_set = set()
    records = []
    
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=['normalized_smiles', 'precursor_mz', 'ionization_mode'])
        df_rg = tbl.to_pandas()
        for _, row in df_rg.iterrows():
            s = row['normalized_smiles']
            if not isinstance(s, str) or not s or s in unique_smiles_set:
                continue
            unique_smiles_set.add(s)
            
            # Estimate neutral mass from precursor or RDKit
            mode = row.get('ionization_mode', 'positive')
            prec = float(row.get('precursor_mz', 0.0))
            if mode == 'positive':
                neutral_mass = prec - PROTON_MASS
            else:
                neutral_mass = prec + PROTON_MASS
                
            records.append((neutral_mass, s))
            if len(records) >= max_records:
                break
        if len(records) >= max_records:
            break

    print(f"Found {len(records)} unique candidate structures. Computing fingerprints...")
    records.sort(key=lambda x: x[0]) # Pre-sort strictly by neutral mass for binary search
    
    masses = np.array([r[0] for r in records], dtype=np.float32)
    smiles_list = [r[1] for r in records]
    
    # Precompute 2048-bit Morgan fingerprints
    fps = np.zeros((len(records), 2048), dtype=np.float32)
    for idx, s in enumerate(smiles_list):
        fps[idx] = smiles_to_morgan_fingerprint(s, n_bits=2048)
    
    assert fps.sum() > 0, "ERROR: Candidate index fingerprints are all zero! Check RDKit installation."

    # Cache index to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, masses=masses, fps=fps, smiles=np.array(smiles_list, dtype=object))
    print(f"[OK] Candidate index built and saved to {save_path}.")
    return masses, fps, smiles_list

def retrieve_top_k_candidates(
    target_mass: float,
    spec_ret_embed: torch.Tensor,
    spec_fp_logits: torch.Tensor,
    candidate_masses: np.ndarray,
    candidate_fps: np.ndarray,
    candidate_smiles: List[str],
    model: SpecNeuralOperatorNet,
    top_k: int = 25,
    ppm_tolerance: float = 20.0,
    device: torch.device = torch.device('cuda')
) -> List[str]:
    """
    Fast binary search retrieval within mass window + hybrid neural metric ranking.
    """
    tol_da = target_mass * (ppm_tolerance * 1e-6)
    idx_left = np.searchsorted(candidate_masses, target_mass - tol_da)
    idx_right = np.searchsorted(candidate_masses, target_mass + tol_da)

    # If too few candidates within 20 ppm, expand window to 100 ppm
    if (idx_right - idx_left) < top_k:
        tol_da_expanded = target_mass * (100.0 * 1e-6)
        idx_left = np.searchsorted(candidate_masses, target_mass - tol_da_expanded)
        idx_right = np.searchsorted(candidate_masses, target_mass + tol_da_expanded)

    # If still fewer than top_k, take nearest neighbors by mass
    if (idx_right - idx_left) < top_k:
        center_idx = np.searchsorted(candidate_masses, target_mass)
        idx_left = max(0, center_idx - top_k)
        idx_right = min(len(candidate_masses), idx_left + top_k * 2)

    cand_sub_fps = torch.tensor(candidate_fps[idx_left:idx_right], dtype=torch.float32, device=device)
    cand_sub_smiles = candidate_smiles[idx_left:idx_right]

    if len(cand_sub_smiles) == 0:
        return ["CCO"] * top_k

    # 1. Metric Retrieval Cosine Similarity (De Waele et al., ICML 2026)
    with torch.no_grad():
        cand_sub_embeds = model.project_candidate_fingerprint(cand_sub_fps)
        sim_ret = torch.matmul(cand_sub_embeds, spec_ret_embed.squeeze(0)).cpu().numpy()

        # 2. Soft Tanimoto Fingerprint Similarity
        spec_probs = torch.sigmoid(spec_fp_logits).squeeze(0)
        intersection = torch.sum(cand_sub_fps * spec_probs, dim=-1)
        union = torch.sum(cand_sub_fps + spec_probs - (cand_sub_fps * spec_probs), dim=-1)
        sim_tani = (intersection / (union + 1e-6)).cpu().numpy()

    # Hybrid ranking score
    hybrid_scores = 0.6 * sim_ret + 0.4 * sim_tani
    ranked_indices = np.argsort(hybrid_scores)[::-1]

    # Select unique top_k SMILES
    selected = []
    seen = set()
    for idx in ranked_indices:
        s = cand_sub_smiles[idx]
        if s not in seen:
            seen.add(s)
            selected.append(s)
            if len(selected) == top_k:
                break

    # If still fewer than top_k, pad with first candidates
    while len(selected) < top_k:
        selected.append(selected[0] if len(selected) > 0 else "CCO")

    return selected[:top_k]
