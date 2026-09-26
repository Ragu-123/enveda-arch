"""
End-to-End Submission Generation Pipeline
Executes SpecNeuralOperatorNet inference on test spectra and ranks candidates via fast binary search.
Outputs valid /kaggle/working/submission.csv matching exact CASMI MAP@25 format.
"""

import os
import time
import numpy as np
import pandas as pd
import torch

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.retrieval import build_candidate_index_from_train, retrieve_top_k_candidates, PROTON_MASS

def run_submission_pipeline():
    print("=" * 65)
    print("ENVEDA-ARCH: CANDIDATE RETRIEVAL & SUBMISSION PIPELINE")
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

    # 2. Build or load Candidate Index from train.parquet
    train_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
    cand_masses, cand_fps, cand_smiles = build_candidate_index_from_train(
        train_path,
        max_records=150000,
        save_path="/kaggle/working/candidate_index.npz"
    )

    # 3. Load Test Data & Sample Submission
    test_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/test.parquet"
    sample_sub_path = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/sample_submission.csv"

    print("\nLoading test.parquet and sample_submission.csv...")
    df_test = pd.read_parquet(test_path)
    df_sub = pd.read_csv(sample_sub_path)
    
    unique_mol_ids = df_sub['molecule_id'].unique()
    print(f"Target molecules to predict: {len(unique_mol_ids)}")

    # Group test spectra by molecule_id
    grouped_test = df_test.groupby('molecule_id')

    submission_rows = []
    t0 = time.time()

    for idx, mol_id in enumerate(unique_mol_ids):
        if mol_id not in grouped_test.groups:
            # Fallback if molecule missing
            submission_rows.append({"molecule_id": mol_id, "smiles": ";".join(["CCO"] * 25)})
            continue

        mol_spectra = grouped_test.get_group(mol_id)
        
        # Pick the most informative spectrum (highest number of peaks or 40 eV)
        mol_spectra['num_peaks'] = mol_spectra['ms2_mzs'].apply(lambda x: len(x) if x is not None else 0)
        best_spec_row = mol_spectra.sort_values(by='num_peaks', ascending=False).iloc[0]

        # Extract features
        mzs = np.array(best_spec_row['ms2_mzs'], dtype=np.float32)
        ints = np.array(best_spec_row['ms2_normalized_intensities'], dtype=np.float32)
        
        # Select top 128 peaks
        if len(mzs) > 128:
            top_idx = np.argsort(ints)[-128:]
            sorted_idx = top_idx[np.argsort(mzs[top_idx])]
            mzs = mzs[sorted_idx]
            ints = ints[sorted_idx]
            
        cur_len = len(mzs)
        pad_len = 128 - cur_len
        if pad_len > 0:
            padded_mzs = np.pad(mzs, (0, pad_len), constant_values=0.0)
            padded_ints = np.pad(ints, (0, pad_len), constant_values=0.0)
            mask = np.pad(np.ones(cur_len, dtype=bool), (0, pad_len), constant_values=False)
        else:
            padded_mzs = mzs
            padded_ints = ints
            mask = np.ones(128, dtype=bool)

        prec_mz = float(best_spec_row.get('precursor_mz', 0.0))
        ce = best_spec_row.get('collision_energy_ev', 30.0)
        if isinstance(ce, (list, np.ndarray)) and len(ce) > 0:
            ce_val = float(ce[0])
        elif isinstance(ce, (int, float)):
            ce_val = float(ce)
        else:
            ce_val = 30.0

        mode_str = best_spec_row.get('ionization_mode', 'positive')
        mode_val = 1.0 if mode_str == 'positive' else 0.0

        # Exact neutral mass calculation
        if mode_str == 'positive':
            target_neutral_mass = prec_mz - PROTON_MASS
        else:
            target_neutral_mass = prec_mz + PROTON_MASS

        # Forward pass through model
        t_mzs = torch.tensor(padded_mzs, dtype=torch.float32, device=device).unsqueeze(0)
        t_ints = torch.tensor(padded_ints, dtype=torch.float32, device=device).unsqueeze(0)
        t_prec = torch.tensor([[prec_mz]], dtype=torch.float32, device=device)
        t_ce = torch.tensor([[ce_val]], dtype=torch.float32, device=device)
        t_mode = torch.tensor([[mode_val]], dtype=torch.float32, device=device)
        t_mask = torch.tensor(mask, dtype=torch.bool, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model(
                mzs=t_mzs,
                intensities=t_ints,
                precursor_mz=t_prec,
                collision_energy=t_ce,
                mode=t_mode,
                mask=t_mask
            )
            spec_ret_embed = outputs["retrieval_embedding"]
            spec_fp_logits = outputs["fingerprint_logits"]

        # Fast binary search candidate retrieval & ranking
        top_25_smiles = retrieve_top_k_candidates(
            target_mass=target_neutral_mass,
            spec_ret_embed=spec_ret_embed,
            spec_fp_logits=spec_fp_logits,
            candidate_masses=cand_masses,
            candidate_fps=cand_fps,
            candidate_smiles=cand_smiles,
            model=model,
            top_k=25,
            ppm_tolerance=20.0,
            device=device
        )

        formatted_smiles = ";".join(top_25_smiles)
        submission_rows.append({"molecule_id": mol_id, "smiles": formatted_smiles})

        if (idx + 1) % 50 == 0 or (idx + 1) == len(unique_mol_ids):
            elapsed = time.time() - t0
            print(f"  Processed [{idx+1:3d}/{len(unique_mol_ids):3d}] molecules ({elapsed:.1f}s)")

    # 4. Save and Verify submission.csv
    submission_df = pd.DataFrame(submission_rows)
    out_sub_path = "/kaggle/working/submission.csv"
    submission_df.to_csv(out_sub_path, index=False)
    print(f"\n[SUCCESS] Generated submission with {len(submission_df)} rows saved to: {out_sub_path}")

    # Strict Validation against sample_submission.csv
    assert len(submission_df) == len(df_sub), f"Row count mismatch! {len(submission_df)} vs {len(df_sub)}"
    assert list(submission_df.columns) == ['molecule_id', 'smiles'], f"Columns mismatch! {submission_df.columns}"
    assert submission_df['smiles'].isna().sum() == 0, "Contains NaNs in SMILES!"

    # Verify each row has exactly 25 candidate SMILES
    counts = submission_df['smiles'].apply(lambda s: len(s.split(';')))
    assert (counts == 25).all(), f"Some rows do not have exactly 25 candidates! Min: {counts.min()}, Max: {counts.max()}"
    print(f"[VERIFIED] All 400 rows strictly contain exactly 25 semicolon-separated candidate SMILES.")

if __name__ == "__main__":
    run_submission_pipeline()
