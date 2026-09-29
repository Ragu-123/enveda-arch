"""
Fast Binary-Search Candidate Retrieval & Metric Ranking Engine
Given trained SpecNeuralOperatorNet, retrieves candidate molecules from pre-sorted neutral mass index
and ranks top 25 candidates using hybrid InfoNCE metric embedding + Morgan fingerprint similarity + Bayes likelihood.
Includes strict InChIKey14 tautomer deduplication and exact adduct mass deconvolution.
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Set
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

# Exact high-precision adduct monoisotopic mass offsets
# neutral_mass = precursor_mz + offset
ADDUCT_MASS_OFFSETS = {
    '[M+H]+': -1.007276,
    '[M+Na]+': -22.989218,
    '[M+NH4]+': -18.033826,
    '[M+K]+': -38.963158,
    '[M-H]-': +1.007276,
    '[M+CH2O2-H]-': -44.998201,  # formate adduct (loss of H + addition of HCOO-)
    '[M+Cl]-': -34.969402,
}

def get_neutral_mass_from_adduct(precursor_mz: float, adduct: Optional[str], mode: str) -> float:
    """Computes exact neutral mass from precursor m/z, adduct, and ionization mode."""
    if adduct and adduct in ADDUCT_MASS_OFFSETS:
        return precursor_mz + ADDUCT_MASS_OFFSETS[adduct]
    if mode == 'positive':
        return precursor_mz - PROTON_MASS
    else:
        return precursor_mz + PROTON_MASS

def compute_inchikey14(smiles: str) -> str:
    """Computes first 14 chars of InChIKey for 2D connectivity deduplication."""
    if not HAS_RDKIT:
        return smiles[:14]
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles[:14]
        return Chem.MolToInchiKey(mol)[:14]
    except Exception:
        return smiles[:14]

def build_candidate_index_from_train(
    train_parquet_path: str,
    max_records: int = 250000,
    save_path: str = "/kaggle/working/candidate_index.npz"
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Builds a pre-sorted candidate database of (neutral_masses, fingerprints, smiles, inchikey14).
    Uses PyArrow row-group streaming to prevent RAM exhaustion.
    Deduplicates candidates by InChIKey14 (preserving the highest quality canonical SMILES).
    """
    if os.path.exists(save_path):
        print(f"Loading cached candidate index from {save_path}...")
        data = np.load(save_path, allow_pickle=True)
        fps = data["fps"]
        if fps.sum() > 0 and "ik14" in data.files:
            ik14_list = list(data["ik14"])
            print(f"[OK] Loaded valid candidate index ({len(data['masses'])} candidates, mean active bits: {fps.sum(axis=1).mean():.1f}).")
            return data["masses"], fps, list(data["smiles"]), ik14_list
        else:
            print(f"[WARNING] Cached candidate index at {save_path} needs rebuild...")

    print(f"Building candidate index from {train_parquet_path}...")
    pf = pq.ParquetFile(train_parquet_path)
    
    unique_ik14_set = set()
    records = []
    
    read_cols = ['normalized_smiles', 'precursor_mz', 'ionization_mode']
    # Check if inchikey14 is present in parquet schema
    schema_names = pf.schema.names
    has_ik14_col = 'inchikey14' in schema_names
    has_adduct_col = 'adduct' in schema_names
    if has_ik14_col:
        read_cols.append('inchikey14')
    if has_adduct_col:
        read_cols.append('adduct')
        
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=read_cols)
        df_rg = tbl.to_pandas()
        for _, row in df_rg.iterrows():
            s = row['normalized_smiles']
            if not isinstance(s, str) or not s:
                continue
                
            ik14 = row.get('inchikey14', None) if has_ik14_col else None
            if not isinstance(ik14, str) or len(ik14) < 14:
                ik14 = compute_inchikey14(s)
                
            if ik14 in unique_ik14_set:
                continue
            unique_ik14_set.add(ik14)
            
            mode = row.get('ionization_mode', 'positive')
            prec = float(row.get('precursor_mz', 0.0))
            adduct = row.get('adduct', None) if has_adduct_col else None
            neutral_mass = get_neutral_mass_from_adduct(prec, adduct, mode)
                
            records.append((neutral_mass, s, ik14))
            if len(records) >= max_records:
                break
        if len(records) >= max_records:
            break

    print(f"Found {len(records)} unique 2D candidate skeletons. Computing Morgan fingerprints...")
    records.sort(key=lambda x: x[0])  # Pre-sort strictly by neutral mass for fast binary search
    
    masses = np.array([r[0] for r in records], dtype=np.float32)
    smiles_list = [r[1] for r in records]
    ik14_list = [r[2] for r in records]
    
    # Precompute 2048-bit Morgan fingerprints
    fps = np.zeros((len(records), 2048), dtype=np.float32)
    for idx, s in enumerate(smiles_list):
        fps[idx] = smiles_to_morgan_fingerprint(s, n_bits=2048)
    
    assert fps.sum() > 0, "ERROR: Candidate index fingerprints are all zero! Check RDKit installation."

    # Cache index to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        masses=masses,
        fps=fps,
        smiles=np.array(smiles_list, dtype=object),
        ik14=np.array(ik14_list, dtype=object)
    )
    print(f"[OK] Candidate index built and saved to {save_path} ({len(records)} unique 2D skeletons).")
    return masses, fps, smiles_list, ik14_list

def retrieve_top_k_candidates(
    target_mass: float,
    spec_ret_embed: torch.Tensor,
    spec_fp_logits: torch.Tensor,
    candidate_masses: np.ndarray,
    candidate_fps: np.ndarray,
    candidate_smiles: List[str],
    candidate_ik14: Optional[List[str]] = None,
    model: Optional[SpecNeuralOperatorNet] = None,
    top_k: int = 25,
    ppm_tolerance: float = 10.0,
    device: torch.device = torch.device('cuda')
) -> List[str]:
    """
    Fast binary search retrieval within mass window + multi-evidence ranking + InChIKey14 deduplication.
    Combines:
    1. Continuous metric cosine similarity in S^2047 (De Waele et al. ICML 2026)
    2. Vectorized Bernoulli log-likelihood (haideptry Top 1 solution: Score = f_c * z)
    3. Soft Tanimoto IoU similarity
    """
    tol_da = target_mass * (ppm_tolerance * 1e-6)
    idx_left = np.searchsorted(candidate_masses, target_mass - tol_da)
    idx_right = np.searchsorted(candidate_masses, target_mass + tol_da)

    # If too few candidates within ppm_tolerance, expand to 25 ppm, then 100 ppm
    if (idx_right - idx_left) < top_k:
        tol_da_expanded = target_mass * (25.0 * 1e-6)
        idx_left = np.searchsorted(candidate_masses, target_mass - tol_da_expanded)
        idx_right = np.searchsorted(candidate_masses, target_mass + tol_da_expanded)

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
    cand_sub_ik14 = candidate_ik14[idx_left:idx_right] if candidate_ik14 is not None else None

    if len(cand_sub_smiles) == 0:
        return ["CCO"] * top_k

    with torch.no_grad():
        # 1. Metric Retrieval Cosine Similarity on Hypersphere S^2047
        cand_sub_embeds = F.normalize(cand_sub_fps, p=2, dim=-1)
        ret_query = F.normalize(spec_ret_embed.view(1, -1), p=2, dim=-1)
        sim_ret = torch.matmul(cand_sub_embeds, ret_query.squeeze(0)).cpu().numpy()

        # 2. Vectorized Bernoulli Log-Likelihood (Score = f_c * z)
        # Proven by haideptry (Top 1) to be exact Bayes posterior ranking under independent Bernoulli bits
        z_query = spec_fp_logits.view(-1)
        bayes_scores = torch.matmul(cand_sub_fps, z_query).cpu().numpy()
        # Min-max scale bayes_scores across the window to [0, 1]
        b_min, b_max = bayes_scores.min(), bayes_scores.max()
        bayes_norm = (bayes_scores - b_min) / (b_max - b_min + 1e-6)

        # 3. Soft Tanimoto Overlap
        spec_probs = torch.sigmoid(z_query)
        intersection = torch.sum(cand_sub_fps * spec_probs, dim=-1)
        union = torch.sum(cand_sub_fps + spec_probs - (cand_sub_fps * spec_probs), dim=-1)
        sim_tani = (intersection / (union + 1e-6)).cpu().numpy()

    # Unified Multi-Evidence Ranking Score:
    # 40% Contrastive Metric Embedding + 35% Bayes Neural Likelihood + 25% Soft Tanimoto
    unified_scores = 0.40 * sim_ret + 0.35 * bayes_norm + 0.25 * sim_tani
    ranked_indices = np.argsort(unified_scores)[::-1]

    # Select top_k candidates with STRICT InChIKey14 deduplication (Udam Liyanage #743254)
    selected = []
    seen_ik14: Set[str] = set()
    seen_smiles: Set[str] = set()

    for idx in ranked_indices:
        s = cand_sub_smiles[idx]
        ik14 = cand_sub_ik14[idx] if cand_sub_ik14 is not None else compute_inchikey14(s)

        if ik14 not in seen_ik14 and s not in seen_smiles:
            seen_ik14.add(ik14)
            seen_smiles.add(s)
            selected.append(s)
            if len(selected) == top_k:
                break

    # If still fewer than top_k due to strict deduplication, pad with best remaining SMILES
    for idx in ranked_indices:
        if len(selected) >= top_k:
            break
        s = cand_sub_smiles[idx]
        if s not in seen_smiles:
            seen_smiles.add(s)
            selected.append(s)

    while len(selected) < top_k:
        selected.append(selected[0] if len(selected) > 0 else "CCO")

    return selected[:top_k]
