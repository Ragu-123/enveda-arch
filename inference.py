"""
Enveda CASMI 2026: End-to-End Test Set Inference & Submission Generator
Pipeline:
1. Multi-Spectrum Multi-Energy Fusion (20 eV, 40 eV, stepped eV, +/- polarities)
2. Sub-ppm consensus neutral mass deconvolution
3. Precursor-windowed candidate retrieval from unified index (+-8.5 ppm)
4. Vectorized InfoNCE Metric Cosine Similarity + Bayes Bernoulli Likelihood + Soft Tanimoto IoU
5. MetFrag-lite In-Silico Bond Cleavage Explainability Kernel
6. Shortlist Tautomer Canonicalization & Strict InChIKey14 Deduplication
7. Validates and writes exactly 400 rows (25 ranked candidate SMILES per row) to submission.csv
"""

import os
import sys
import glob
import time
import argparse
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from enveda_arch.models.neural_operator_net import SpecNeuralOperatorNet
from enveda_arch.retrieval import (
    build_candidate_index_from_train,
    retrieve_top_k_candidates,
    get_neutral_mass_from_adduct,
    compute_inchikey14,
    canon_inchikey14,
    PROTON_MASS
)
from enveda_arch.library_search import clean_peaks, MassBankLibrary

def clean_and_pad_spectrum(
    mzs_raw: list, ints_raw: list, max_peaks: int = 128, min_rel_int: float = 0.005
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Cleans raw timsTOF spectrum:
    - Drops noise peaks below min_rel_int (0.5% base peak)
    - Selects top max_peaks by intensity and sorts by m/z
    - Pads with zeros to max_peaks
    """
    if mzs_raw is None or len(mzs_raw) == 0:
        return np.zeros(max_peaks, dtype=np.float32), np.zeros(max_peaks, dtype=np.float32), np.zeros(max_peaks, dtype=bool)

    mzs = np.array(mzs_raw, dtype=np.float32)
    ints = np.array(ints_raw, dtype=np.float32)

    base_int = ints.max() if len(ints) > 0 else 1.0
    if base_int > 0:
        norm_ints = ints / base_int
        valid = norm_ints >= min_rel_int
        if valid.sum() >= 3:
            mzs = mzs[valid]
            ints = norm_ints[valid]
        else:
            ints = norm_ints

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

def find_checkpoint(explicit_path: Optional[str] = None) -> str:
    """Finds best available trained checkpoint with automatic fallback discovery."""
    if explicit_path and os.path.exists(explicit_path):
        return explicit_path

    search_patterns = [
        "/kaggle/working/checkpoints/spec_neural_operator_best.pt",
        "/kaggle/working/checkpoints/spec_neural_operator.pt",
        "/kaggle/input/**/spec_neural_operator_best.pt",
        "/kaggle/input/**/spec_neural_operator.pt",
        "checkpoints/spec_neural_operator_best.pt",
        "checkpoints/spec_neural_operator.pt",
    ]
    for pattern in search_patterns:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return sorted(matches)[0]

    raise FileNotFoundError(
        "Could not find any SpecNeuralOperatorNet checkpoint! "
        "Please provide --checkpoint <path_to_model.pt>"
    )

def find_or_build_candidate_index(
    index_path: Optional[str] = None,
    train_parquet_path: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/train.parquet"
) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Loads pre-built candidate index, or builds it from train.parquet if missing."""
    candidates_to_check = [
        index_path,
        "/kaggle/working/candidate_index_merged.npz",
        "/kaggle/working/candidate_index.npz",
        "candidate_index_merged.npz",
        "candidate_index.npz"
    ]
    for p in candidates_to_check:
        if p and os.path.exists(p):
            print(f"[OK] Loading candidate index from: {p}")
            data = np.load(p, allow_pickle=True)
            return (
                data["masses"],
                data["fps"],
                list(data["smiles"]),
                list(data["ik14"]) if "ik14" in data.files else None
            )

    print(f"[INIT] Precomputed candidate index not found. Building from {train_parquet_path}...")
    save_path = "/kaggle/working/candidate_index.npz"
    masses, fps, smiles, ik14 = build_candidate_index_from_train(
        train_parquet_path=train_parquet_path,
        max_records=250000,
        save_path=save_path
    )
    return masses, fps, smiles, ik14

def run_inference(
    checkpoint_path: Optional[str] = None,
    test_parquet: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/test.parquet",
    sample_sub: str = "/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/sample_submission.csv",
    candidate_index_path: Optional[str] = None,
    output_csv: str = "/kaggle/working/submission.csv",
    massbank_path: Optional[str] = None,
    ppm_tolerance: float = 10.0,
    top_k: int = 25
):
    print("=" * 75)
    print("ENVEDA CASMI 2026: END-TO-END TEST SET INFERENCE PIPELINE")
    print("=" * 75)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Load Trained SpecNeuralOperatorNet
    ckpt_file = find_checkpoint(checkpoint_path)
    print(f"[OK] Checkpoint found: {ckpt_file}")

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
    state = torch.load(ckpt_file, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()
    print("[OK] Loaded trained weights into SpecNeuralOperatorNet.")

    # 2. Candidate Index
    cand_masses, cand_fps, cand_smiles, cand_ik14 = find_or_build_candidate_index(
        index_path=candidate_index_path
    )
    print(f"[OK] Candidate database ready: {len(cand_masses):,} structures loaded.")

    # 3. Optional Reference Library
    mb_lib = None
    if massbank_path and os.path.exists(massbank_path):
        try:
            print(f"[OK] Ingesting MassBank Reference Library from {massbank_path}...")
            mb_lib = MassBankLibrary(massbank_path)
        except Exception as e:
            print(f"  Note on MassBank loading: {e}")

    # 4. Load Test Spectra
    if not os.path.exists(test_parquet):
        raise FileNotFoundError(f"Test dataset not found at: {test_parquet}")
    if not os.path.exists(sample_sub):
        raise FileNotFoundError(f"Sample submission not found at: {sample_sub}")

    print(f"\nLoading test spectra from {test_parquet}...")
    df_test = pd.read_parquet(test_parquet)
    df_sub = pd.read_csv(sample_sub)
    unique_mol_ids = df_sub['molecule_id'].unique()
    print(f"Total target molecules to predict: {len(unique_mol_ids):,}", flush=True)

    grouped_test = df_test.groupby('molecule_id')
    submission_rows = []
    t0 = time.time()

    for idx, mol_id in enumerate(unique_mol_ids):
        if mol_id not in grouped_test.groups:
            submission_rows.append({"molecule_id": mol_id, "smiles": ";".join(["CCO"] * top_k)})
            continue

        mol_spectra = grouped_test.get_group(mol_id)
        all_ret_embeds = []
        all_fp_logits = []
        estimated_masses = []
        all_adducts = []
        merged_mzs = []
        merged_ints = []

        for _, spec_row in mol_spectra.iterrows():
            mzs_raw = spec_row.get('ms2_mzs')
            ints_raw = spec_row.get('ms2_normalized_intensities')
            pad_mzs, pad_ints, mask = clean_and_pad_spectrum(mzs_raw, ints_raw, max_peaks=128, min_rel_int=0.005)

            if mask.any():
                merged_mzs.extend(list(pad_mzs[mask]))
                merged_ints.extend(list(pad_ints[mask]))

            prec_mz = float(spec_row.get('precursor_mz', 0.0))
            adduct = spec_row.get('adduct', None)
            if adduct:
                all_adducts.append(str(adduct))
            mode_str = spec_row.get('ionization_mode', 'positive')
            mode_val = 1.0 if mode_str == 'positive' else 0.0

            ce = spec_row.get('collision_energy_ev', 30.0)
            if isinstance(ce, (list, np.ndarray)) and len(ce) > 0:
                ce_val = abs(float(ce[0]))
            elif isinstance(ce, (int, float)):
                ce_val = abs(float(ce))
            else:
                ce_val = 30.0

            neutral_mass = get_neutral_mass_from_adduct(prec_mz, adduct, mode_str)
            estimated_masses.append(neutral_mass)

            t_mzs = torch.tensor(pad_mzs, dtype=torch.float32, device=device).unsqueeze(0)
            t_ints = torch.tensor(pad_ints, dtype=torch.float32, device=device).unsqueeze(0)
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
                all_ret_embeds.append(out["retrieval_embedding"].float())
                all_fp_logits.append(out["fingerprint_logits"].float())

        # Consensus multi-energy representations
        fused_neutral_mass = float(np.median(estimated_masses))
        fused_fp_logits = torch.stack(all_fp_logits, dim=0).mean(dim=0).squeeze(0)
        fused_ret_embed = F.normalize(torch.stack(all_ret_embeds, dim=0).mean(dim=0), p=2, dim=-1)

        # Informative query fragments for cleavage kernel
        if len(merged_mzs) > 0:
            m_arr = np.array(merged_mzs, dtype=np.float32)
            i_arr = np.array(merged_ints, dtype=np.float32)
            if len(m_arr) > 48:
                top_idx = np.argsort(i_arr)[-48:]
                m_arr, i_arr = m_arr[top_idx], i_arr[top_idx]
            s_idx = np.argsort(m_arr)
            query_peaks = (m_arr[s_idx], i_arr[s_idx])
        else:
            query_peaks = None

        consensus_adduct = max(set(all_adducts), key=all_adducts.count) if all_adducts else "[M+H]+"

        # Direct Library Match Gate
        lib_match_smiles = None
        lib_match_sim = 0.0
        if mb_lib and query_peaks is not None:
            lib_res = mb_lib.query(
                prec_mz=fused_neutral_mass + PROTON_MASS,
                neutral_mass=fused_neutral_mass,
                mode_str="positive",
                qmz=query_peaks[0],
                qit=query_peaks[1],
                tol_ppm=ppm_tolerance,
                min_sim=0.85
            )
            if lib_res:
                lib_match_smiles = lib_res[0][0]
                lib_match_sim = lib_res[0][1]

        # Candidate retrieval & metric ranking
        top_candidates = retrieve_top_k_candidates(
            target_mass=fused_neutral_mass,
            spec_ret_embed=fused_ret_embed,
            spec_fp_logits=fused_fp_logits,
            candidate_masses=cand_masses,
            candidate_fps=cand_fps,
            candidate_smiles=cand_smiles,
            candidate_ik14=cand_ik14,
            query_peaks=query_peaks,
            adduct=consensus_adduct,
            library_match_smiles=lib_match_smiles,
            library_match_sim=lib_match_sim,
            model=model,
            top_k=top_k,
            ppm_tolerance=ppm_tolerance,
            device=device
        )

        formatted_smiles = ";".join(top_candidates)
        submission_rows.append({"molecule_id": mol_id, "smiles": formatted_smiles})

        if (idx + 1) % 25 == 0 or (idx + 1) == len(unique_mol_ids):
            elapsed = time.time() - t0
            rate = (idx + 1) / max(0.1, elapsed)
            print(f"  Processed [{idx+1:3d}/{len(unique_mol_ids):3d}] molecules ({elapsed:.1f}s, {rate:.1f} mols/s)", flush=True)

    # 5. Save and Validate Output CSV
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    out_df = pd.DataFrame(submission_rows)
    out_df.to_csv(output_csv, index=False)

    print("\n" + "=" * 65)
    print("SUBMISSION VERIFICATION CHECKS")
    print("=" * 65)
    print(f"Output saved to: {output_csv}")
    print(f"Total rows: {len(out_df)} (Expected: {len(unique_mol_ids)})")
    assert len(out_df) == len(unique_mol_ids), f"Row count mismatch! Got {len(out_df)}, expected {len(unique_mol_ids)}"

    counts = [len(s.split(';')) for s in out_df['smiles']]
    assert all(c == top_k for c in counts), f"Mismatch in candidate count! Min: {min(counts)}, Max: {max(counts)}"
    print(f"[OK] Exactly {top_k} candidate SMILES per row for all {len(out_df)} molecules.")

    assert out_df['smiles'].isna().sum() == 0, "Null values found in smiles column!"
    assert all(len(s.strip()) > 0 for s in out_df['smiles']), "Blank entries found!"
    print("[OK] Zero nulls or blank entries detected.")

    print(f"\nSample Prediction (molecule_id: {out_df.iloc[0]['molecule_id']}):")
    for r, s in enumerate(out_df.iloc[0]['smiles'].split(';')[:5], start=1):
        print(f"  Rank {r:2d}: {s}")
    print("=" * 65)
    print(f"[SUCCESS] Inference complete! Ready for submission: {output_csv}")

def main():
    parser = argparse.ArgumentParser(description="Enveda CASMI 2026 Inference Pipeline")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to spec_neural_operator checkpoint (.pt)")
    parser.add_argument("--test-parquet", type=str, default="/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/test.parquet")
    parser.add_argument("--sample-submission", type=str, default="/kaggle/input/competitions/enveda-CASMI26-molecule-id-mass-spectra/sample_submission.csv")
    parser.add_argument("--candidate-index", type=str, default=None, help="Path to precomputed candidate_index.npz")
    parser.add_argument("--massbank-library", type=str, default="/kaggle/input/datasets/samartalwar/casmi-2026-spectral-library-massbankharmonized/spectra.parquet")
    parser.add_argument("--output-csv", type=str, default="/kaggle/working/submission.csv")
    parser.add_argument("--ppm-tolerance", type=float, default=10.0)
    parser.add_argument("--top-k", type=int, default=25)
    args = parser.parse_args()

    run_inference(
        checkpoint_path=args.checkpoint,
        test_parquet=args.test_parquet,
        sample_sub=args.sample_submission,
        candidate_index_path=args.candidate_index,
        output_csv=args.output_csv,
        massbank_path=args.massbank_library,
        ppm_tolerance=args.ppm_tolerance,
        top_k=args.top_k
    )

if __name__ == "__main__":
    main()
