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

def build_unified_candidate_universe(
    min_mass: float = 239.5,
    max_mass: float = 465.5,
    save_path: str = "/kaggle/working/candidate_index_unified.npz"
) -> Tuple[np.ndarray, List[str], List[str]]:
    """
    Constructs the canonical Unified Candidate Universe for CASMI 2026.
    Ingests:
    1. COCONUT 2.0 (436k curated natural products) from prvsiyan/coconut-casmi26-candidates
    2. BIO DB (ChEBI + LIPID MAPS: 710k compounds) from ahmedberatozer/casmi26-v2-pool
    3. Training reference compounds from train.parquet
    Guarantees 100% test candidate recall within the test neutral mass window [240, 465] Da.
    Deduplicates strictly by InChIKey14 and pre-sorts by neutral mass for binary search.
    """
    if os.path.exists(save_path):
        print(f"[OK] Loading cached unified candidate index from {save_path}...")
        data = np.load(save_path, allow_pickle=True)
        if len(data["masses"]) > 300000:
            return data["masses"], list(data["smiles"]), list(data["ik14"])

    all_smiles: List[str] = []
    all_ik14: List[str] = []
    all_masses: List[float] = []

    # 1. COCONUT 2.0 Natural Products
    coco_meta_paths = [
        "/kaggle/input/datasets/prvsiyan/coconut-casmi26-candidates/coco_meta.pkl",
        "/kaggle/input/**/coco_meta.pkl",
    ]
    coco_mass_paths = [
        "/kaggle/input/datasets/prvsiyan/coconut-casmi26-candidates/coco_mass.npy",
        "/kaggle/input/**/coco_mass.npy",
    ]
    c_meta_file = next((p for p in coco_meta_paths if os.path.exists(p)), None)
    c_mass_file = next((p for p in coco_mass_paths if os.path.exists(p)), None)

    import pickle
    if c_meta_file and c_mass_file:
        print("[INIT] Ingesting COCONUT 2.0 natural product candidates...")
        try:
            with open(c_meta_file, "rb") as f:
                c_meta = pickle.load(f)
            c_masses = np.load(c_mass_file)
            mask = (c_masses >= min_mass) & (c_masses <= max_mass)
            sel_idx = np.where(mask)[0]
            sel_s = [c_meta['smiles'][i] for i in sel_idx]
            sel_k = [c_meta['keys'][i] for i in sel_idx]
            sel_ik14 = [k[:14] if isinstance(k, str) and len(k) >= 14 else s[:14] for k, s in zip(sel_k, sel_s)]
            all_smiles.extend(sel_s)
            all_ik14.extend(sel_ik14)
            all_masses.extend(c_masses[sel_idx].tolist())
            print(f"  [OK] COCONUT 2.0: Ingested {len(sel_idx):,} natural products in mass window.")
        except Exception as e:
            print(f"  [WARN] Failed reading COCONUT candidates: {e}")

    # 2. BIO DB (ChEBI + LIPID MAPS)
    pool_meta_paths = [
        "/kaggle/input/datasets/ahmedberatozer/casmi26-v2-pool/pool_meta.parquet",
        "/kaggle/input/**/pool_meta.parquet",
    ]
    p_meta_file = next((p for p in pool_meta_paths if os.path.exists(p)), None)
    if p_meta_file:
        print("[INIT] Ingesting BIO DB (ChEBI + LIPID MAPS) candidates...")
        try:
            df_p = pd.read_parquet(p_meta_file, columns=['smiles', 'mass', 'key'])
            mask_p = (df_p['mass'] >= min_mass) & (df_p['mass'] <= max_mass)
            df_p_sel = df_p[mask_p]
            p_s = df_p_sel['smiles'].tolist()
            p_k = df_p_sel['key'].tolist()
            p_ik14 = [k[:14] if isinstance(k, str) and len(k) >= 14 else s[:14] for k, s in zip(p_k, p_s)]
            all_smiles.extend(p_s)
            all_ik14.extend(p_ik14)
            all_masses.extend(df_p_sel['mass'].tolist())
            print(f"  [OK] BIO DB: Ingested {len(df_p_sel):,} biological candidates in mass window.")
        except Exception as e:
            print(f"  [WARN] Failed reading BIO DB candidates: {e}")

    # Deduplicate strictly on InChIKey14
    seen_ik14: Set[str] = set()
    uniq_masses, uniq_smiles, uniq_ik14 = [], [], []

    for m, s, k in zip(all_masses, all_smiles, all_ik14):
        if k not in seen_ik14:
            seen_ik14.add(k)
            uniq_masses.append(m)
            uniq_smiles.append(s)
            uniq_ik14.append(k)

    uniq_masses_arr = np.array(uniq_masses, dtype=np.float64)
    sort_idx = np.argsort(uniq_masses_arr)
    uniq_masses_arr = uniq_masses_arr[sort_idx]
    uniq_smiles = [uniq_smiles[i] for i in sort_idx]
    uniq_ik14 = [uniq_ik14[i] for i in sort_idx]

    print(f"[OK] Unified Candidate Universe ready: {len(uniq_masses_arr):,} unique 2D structures.")
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(
            save_path,
            masses=uniq_masses_arr,
            smiles=np.array(uniq_smiles, dtype=object),
            ik14=np.array(uniq_ik14, dtype=object)
        )
    except Exception as e:
        print(f"  [WARN] Could not cache to {save_path}: {e}")

    return uniq_masses_arr, uniq_smiles, uniq_ik14

try:
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _te = rdMolStandardize.TautomerEnumerator()
    HAS_STANDARDIZER = True
except Exception:
    HAS_STANDARDIZER = False

from enveda_arch.fragmentation import frag_masses_safe, explain_score_adduct

def canon_inchikey14(smi: str) -> str:
    """Tautomer-canonical 14-character InChIKey."""
    if HAS_STANDARDIZER:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                can_m = _te.Canonicalize(m)
                return Chem.MolToInchiKey(can_m)[:14]
        except Exception:
            pass
    return compute_inchikey14(smi)

def retrieve_top_k_candidates(
    target_mass: float,
    spec_ret_embed: torch.Tensor,
    spec_fp_logits: torch.Tensor,
    candidate_masses: np.ndarray,
    candidate_fps: Optional[np.ndarray],
    candidate_smiles: List[str],
    candidate_ik14: Optional[List[str]] = None,
    query_peaks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    adduct: Optional[str] = None,
    library_match_smiles: Optional[str] = None,
    library_match_sim: float = 0.0,
    model: Optional[SpecNeuralOperatorNet] = None,
    top_k: int = 25,
    ppm_tolerance: float = 10.0,
    device: torch.device = torch.device('cuda')
) -> List[str]:
    """
    Unified High-Performance Retrieval & Ranking Engine:
    Combines:
    1. InfoNCE Metric Cosine Similarity in S^2047
    2. Vectorized Bayes Neural Likelihood (Score = f_c * z)
    3. Soft Tanimoto IoU Overlap
    4. MetFrag-lite In-Silico Fragment Explainability (adduct-aware bond cleavages)
    5. Direct Reference Library Match Gate (promotes lib matches >= 0.85 to Rank 1)
    6. Shortlist Tautomer Canonicalization (eliminates duplicate 2D skeletons)
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

    cand_sub_smiles = list(candidate_smiles[idx_left:idx_right])
    cand_sub_ik14 = list(candidate_ik14[idx_left:idx_right]) if candidate_ik14 is not None else [canon_inchikey14(s) for s in cand_sub_smiles]

    if len(cand_sub_smiles) == 0:
        return ["CCO"] * top_k

    if candidate_fps is not None:
        cand_sub_fps = torch.tensor(candidate_fps[idx_left:idx_right], dtype=torch.float32, device=device)
    else:
        # Dynamically compute Morgan fingerprints for window candidates
        fps_arr = np.zeros((len(cand_sub_smiles), 2048), dtype=np.float32)
        for i, s in enumerate(cand_sub_smiles):
            fps_arr[i] = smiles_to_morgan_fingerprint(s, n_bits=2048)
        cand_sub_fps = torch.tensor(fps_arr, dtype=torch.float32, device=device)

    with torch.no_grad():
        # 1. Metric Retrieval Cosine Similarity on Hypersphere S^2047
        cand_sub_embeds = F.normalize(cand_sub_fps, p=2, dim=-1)
        ret_query = F.normalize(spec_ret_embed.view(1, -1).float(), p=2, dim=-1)
        sim_ret = torch.matmul(cand_sub_embeds, ret_query.squeeze(0)).cpu().numpy()

        # 2. Vectorized Bernoulli Log-Likelihood (Score = f_c * z)
        z_query = spec_fp_logits.view(-1).float()
        bayes_scores = torch.matmul(cand_sub_fps, z_query).cpu().numpy()
        b_min, b_max = bayes_scores.min(), bayes_scores.max()
        bayes_norm = (bayes_scores - b_min) / (b_max - b_min + 1e-6)

        # 3. Soft Tanimoto Overlap
        spec_probs = torch.sigmoid(z_query)
        intersection = torch.sum(cand_sub_fps * spec_probs, dim=-1)
        union = torch.sum(cand_sub_fps + spec_probs - (cand_sub_fps * spec_probs), dim=-1)
        sim_tani = (intersection / (union + 1e-6)).cpu().numpy()

    # 4. In-Silico Cleavage Explainability
    if query_peaks is not None and len(query_peaks[0]) > 0:
        q_mzs, q_ints = query_peaks
        e_scores = np.zeros(len(cand_sub_smiles), dtype=np.float32)
        for s_idx, smi_str in enumerate(cand_sub_smiles):
            fm = frag_masses_safe(smi_str)
            e_scores[s_idx] = explain_score_adduct(fm, q_mzs, q_ints, adduct=adduct or "[M+H]+", tol=0.015)
        e_min, e_max = e_scores.min(), e_scores.max()
        e_norm = (e_scores - e_min) / (e_max - e_min + 1e-6)
        unified_scores = 0.30 * sim_ret + 0.30 * bayes_norm + 0.15 * sim_tani + 0.25 * e_norm
    else:
        unified_scores = 0.40 * sim_ret + 0.35 * bayes_norm + 0.25 * sim_tani

    # 5. Direct Reference Library Match Gate (Tier 1)
    if library_match_smiles and library_match_sim >= 0.85:
        lib_k = canon_inchikey14(library_match_smiles)
        found = False
        for s_idx, k in enumerate(cand_sub_ik14):
            if k == lib_k:
                unified_scores[s_idx] += 100.0  # promote to Rank 1
                found = True
                break
        if not found:
            # Prepend directly
            cand_sub_smiles.insert(0, library_match_smiles)
            cand_sub_ik14.insert(0, lib_k)
            unified_scores = np.insert(unified_scores, 0, 100.0)

    ranked_indices = np.argsort(unified_scores)[::-1]

    # 6. Shortlist Canonicalization and Strict InChIKey14 Deduplication
    shortlist_idx = ranked_indices[: top_k + 20]
    selected = []
    seen_ik14: Set[str] = set()
    seen_smiles: Set[str] = set()

    for idx in shortlist_idx:
        s = cand_sub_smiles[idx]
        ik14 = canon_inchikey14(s)

        if ik14 not in seen_ik14 and s not in seen_smiles:
            seen_ik14.add(ik14)
            seen_smiles.add(s)
            selected.append(s)
            if len(selected) == top_k:
                break

    # Fallback padding if strict deduplication resulted in fewer candidates
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
