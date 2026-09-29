"""
End-to-End Submission Generation Pipeline
Executes SpecNeuralOperatorNet inference on test spectra with:
- Multi-Spectrum Multi-Energy Fusion (pooling across 20 eV, 40 eV, stepped eV, +/- polarities)
- Sub-ppm consensus neutral mass deconvolution
- Spectral noise cleaning (relative intensity floor 0.5%)
- Strict InChIKey14 candidate deduplication
Outputs valid /kaggle/working/submission.csv matching exact CASMI MAP@25 format.
"""

import os
import time
from typing import List, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.retrieval import (
    build_candidate_index_from_train,
    retrieve_top_k_candidates,
    get_neutral_mass_from_adduct,
    PROTON_MASS
)

def clean_and_pad_spectrum(mzs_raw: list, ints_raw: list, max_peaks: int = 128, min_rel_int: float = 0.005) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cleans raw timsTOF spectrum:
    - Drops noise peaks below min_rel_int (0.5% of base peak) per starkhushi insight
    - Selects top max_peaks by intensity and sorts by m/z
    - Pads with zeros to max_peaks
    """
    if mzs_raw is None or len(mzs_raw) == 0:
        return np.zeros(max_peaks, dtype=np.float32), np.zeros(max_peaks, dtype=np.float32), np.zeros(max_peaks, dtype=bool)

    mzs = np.array(mzs_raw, dtype=np.float32)
    ints = np.array(ints_raw, dtype=np.float32)

    # Relative intensity cleaning
    base_int = ints.max() if len(ints) > 0 else 1.0
    if base_int > 0:
        norm_ints = ints / base_int
        valid = norm_ints >= min_rel_int
        if valid.sum() >= 3:
            mzs = mzs[valid]
            ints = norm_ints[valid]
        else:
            ints = norm_ints

    # Keep top max_peaks
    if len(mzs) > max_peaks:
        top_idx = np.argsort(ints)[-max_peaks:]
        sorted_idx = top_idx[np.argsort(mzs[top_idx])]
        mzs = mzs[sorted_idx]
        ints = ints[sorted_idx]
    else:
        sort_order = np.argsort(mzs)
        mzs = mzs[sort_order]
        ints = ints[sort_order]

    cur_len = len(mzs)
    pad_len = max_peaks - cur_len
    if pad_len > 0:
        padded_mzs = np.pad(mzs, (0, pad_len), constant_values=0.0)
        padded_ints = np.pad(ints, (0, pad_len), constant_values=0.0)
        mask = np.pad(np.ones(cur_len, dtype=bool), (0, pad_len), constant_values=False)
    else:
        padded_mzs = mzs[:max_peaks]
        padded_ints = ints[:max_peaks]
        mask = np.ones(max_peaks, dtype=bool)

    return padded_mzs, padded_ints, mask

def run_submission_pipeline():
    print("=" * 65)
    print("ENVEDA-ARCH: MULTI-SPECTRUM CANDIDATE RETRIEVAL & SUBMISSION PIPELINE")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load trained SpecNeuralOperatorNet (prioritize best validation checkpoint)
    ckpt_path = "/kaggle/working/checkpoints/spec_neural_operator_best.pt"
    if not os.path.exists(ckpt_path):
        ckpt_path = "/kaggle/working/checkpoints/spec_neural_operator.pt"
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}!")
    print(f"Loading weights from checkpoint: {ckpt_path}")

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
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print("[OK] Loaded trained SpecNeuralOperatorNet from checkpoint.")

    # 2. Load Unified Candidate Index (train + COCONUT, 476k structures)
    cand_path = "/kaggle/working/candidate_index_merged.npz"
    if not os.path.exists(cand_path):
        cand_path = "/kaggle/working/candidate_index.npz"
    print(f"Loading candidate index from: {cand_path}", flush=True)
    cand_data = np.load(cand_path, allow_pickle=True)
    cand_masses = cand_data["masses"]
    cand_fps = cand_data["fps"]
    cand_smiles = cand_data["smiles"]
    cand_ik14 = cand_data["ik14"]
    print(f"[OK] Loaded {len(cand_masses):,} unified candidates across {cand_masses.min():.2f} - {cand_masses.max():.2f} Da.", flush=True)

    # 3. Load Test Data & Sample Submission
    test_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"
    sample_sub_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/sample_submission.csv"

    print("\nLoading test.parquet and sample_submission.csv...", flush=True)
    df_test = pd.read_parquet(test_path)
    df_sub = pd.read_csv(sample_sub_path)
    
    unique_mol_ids = df_sub['molecule_id'].unique()
    print(f"Target molecules to predict: {len(unique_mol_ids)}", flush=True)

    grouped_test = df_test.groupby('molecule_id')

    submission_rows = []
    t0 = time.time()

    for idx, mol_id in enumerate(unique_mol_ids):
        if mol_id not in grouped_test.groups:
            # Fallback if molecule missing
            submission_rows.append({"molecule_id": mol_id, "smiles": ";".join(["CCO"] * 25)})
            continue

        mol_spectra = grouped_test.get_group(mol_id)
        
        # Multi-Spectrum Multi-Energy Fusion
        all_ret_embeds = []
        all_fp_logits = []
        estimated_masses = []
        all_adducts = []
        merged_mzs = []
        merged_ints = []

        for _, spec_row in mol_spectra.iterrows():
            mzs_raw = spec_row.get('ms2_mzs')
            ints_raw = spec_row.get('ms2_normalized_intensities')
            padded_mzs, padded_ints, mask = clean_and_pad_spectrum(mzs_raw, ints_raw, max_peaks=128, min_rel_int=0.005)

            # Collect cleaned peaks (filtered >= 0.5% base peak)
            if mask.any():
                merged_mzs.extend(list(padded_mzs[mask]))
                merged_ints.extend(list(padded_ints[mask]))

            prec_mz = float(spec_row.get('precursor_mz', 0.0))
            adduct = spec_row.get('adduct', None)
            if adduct:
                all_adducts.append(adduct)
            mode_str = spec_row.get('ionization_mode', 'positive')
            mode_val = 1.0 if mode_str == 'positive' else 0.0

            # Absolute collision energy (handle negative collision energy in negative mode)
            ce = spec_row.get('collision_energy_ev', 30.0)
            if isinstance(ce, (list, np.ndarray)) and len(ce) > 0:
                ce_val = abs(float(ce[0]))
            elif isinstance(ce, (int, float)):
                ce_val = abs(float(ce))
            else:
                ce_val = 30.0

            neutral_mass = get_neutral_mass_from_adduct(prec_mz, adduct, mode_str)
            estimated_masses.append(neutral_mass)

            t_mzs = torch.tensor(padded_mzs, dtype=torch.float32, device=device).unsqueeze(0)
            t_ints = torch.tensor(padded_ints, dtype=torch.float32, device=device).unsqueeze(0)
            t_prec = torch.tensor([[prec_mz]], dtype=torch.float32, device=device)
            t_ce = torch.tensor([[ce_val]], dtype=torch.float32, device=device)
            t_mode = torch.tensor([[mode_val]], dtype=torch.float32, device=device)
            t_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

            with torch.no_grad():
                out = model(
                    mzs=t_mzs,
                    intensities=t_ints,
                    precursor_mz=t_prec,
                    collision_energy=t_ce,
                    mode=t_mode,
                    mask=t_mask
                )
                all_ret_embeds.append(out["retrieval_embedding"])
                all_fp_logits.append(out["fingerprint_logits"])

        # Consensus representations
        fused_neutral_mass = float(np.median(estimated_masses))
        fused_fp_logits = torch.stack(all_fp_logits, dim=0).mean(dim=0)
        fused_ret_embed = F.normalize(torch.stack(all_ret_embeds, dim=0).mean(dim=0), p=2, dim=-1)

        # Bound merged peaks to top 48 informative fragments for fast cleavage evaluation
        if len(merged_mzs) > 0:
            m_arr = np.array(merged_mzs, dtype=np.float32)
            i_arr = np.array(merged_ints, dtype=np.float32)
            if len(m_arr) > 48:
                top_idx = np.argsort(i_arr)[-48:]
                m_arr = m_arr[top_idx]
                i_arr = i_arr[top_idx]
            s_idx = np.argsort(m_arr)
            query_peaks = (m_arr[s_idx], i_arr[s_idx])
        else:
            query_peaks = None

        consensus_adduct = max(set(all_adducts), key=all_adducts.count) if all_adducts else None

        # Candidate retrieval with InChIKey14 deduplication + MetFrag cleavage explainability
        top_25_smiles = retrieve_top_k_candidates(
            target_mass=fused_neutral_mass,
            spec_ret_embed=fused_ret_embed,
            spec_fp_logits=fused_fp_logits,
            candidate_masses=cand_masses,
            candidate_fps=cand_fps,
            candidate_smiles=cand_smiles,
            candidate_ik14=cand_ik14,
            query_peaks=query_peaks,
            adduct=consensus_adduct,
            model=model,
            top_k=25,
            ppm_tolerance=10.0,
            device=device
        )

        formatted_smiles = ";".join(top_25_smiles)
        submission_rows.append({"molecule_id": mol_id, "smiles": formatted_smiles})

        if (idx + 1) % 25 == 0 or (idx + 1) == len(unique_mol_ids):
            elapsed = time.time() - t0
            print(f"  Processed [{idx+1:3d}/{len(unique_mol_ids):3d}] molecules ({elapsed:.1f}s)", flush=True)

    # 4. Save and Validate Output CSV
    out_df = pd.DataFrame(submission_rows)
    out_path = "/kaggle/working/submission.csv"
    out_df.to_csv(out_path, index=False)

    print("\n" + "=" * 65)
    print("SUBMISSION VERIFICATION CHECKS")
    print("=" * 65)
    print(f"Output saved to: {out_path}")
    print(f"Total rows: {len(out_df)} (Expected: 400)")
    assert len(out_df) == 400, f"Expected 400 rows, got {len(out_df)}!"
    
    # Check 25 SMILES per row
    counts = [len(s.split(';')) for s in out_df['smiles']]
    assert all(c == 25 for c in counts), f"Mismatch in candidate count! Min: {min(counts)}, Max: {max(counts)}"
    print(f"[OK] Exactly 25 candidate SMILES per row for all 400 molecules.")

    # Check for empty or null strings
    assert out_df['smiles'].isna().sum() == 0, "Null values found in smiles column!"
    assert all(len(s.strip()) > 0 for s in out_df['smiles']), "Blank entries found!"
    print("[OK] Zero nulls or blank entries detected.")

    print(f"\nSample Prediction (molecule_id: {out_df.iloc[0]['molecule_id']}):")
    sample_smiles = out_df.iloc[0]['smiles'].split(';')
    for r, s in enumerate(sample_smiles[:5], start=1):
        print(f"  Rank {r:2d}: {s}")
    print("=" * 65)
    print("[SUCCESS] Multi-Spectrum Ensembled Submission generated successfully!")

if __name__ == "__main__":
    run_submission_pipeline()
