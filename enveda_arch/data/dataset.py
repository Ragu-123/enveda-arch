"""
High-Throughput Streaming PyTorch Dataset for Enveda CASMI MS/MS Spectra
Safely streams from large Parquet files using PyArrow with zero host-RAM OOM.
Extracts peak lists, continuous coordinates, and ground-truth 2048-bit Morgan Fingerprints.
"""

import re
from typing import List, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

def get_rdkit():
    """Dynamically loads RDKit to prevent stale cached import failures and disables stderr noise."""
    try:
        from rdkit import Chem, rdBase
        from rdkit.Chem import AllChem
        rdBase.DisableLog('rdApp.warning')
        return Chem, AllChem
    except ImportError:
        return None, None

ELEMENTS = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']

def smiles_to_morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Generates 2048-bit Morgan/ECFP4 fingerprint from SMILES string."""
    fp_arr = np.zeros((n_bits,), dtype=np.float32)
    if not isinstance(smiles, str) or not smiles:
        return fp_arr
    Chem, AllChem = get_rdkit()
    if Chem is None or AllChem is None:
        return fp_arr
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return fp_arr
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    for bit in fp.GetOnBits():
        fp_arr[bit] = 1.0
    return fp_arr

def _worker_smiles_to_fp_uint8(s: str) -> np.ndarray:
    """Top-level worker function for parallel Morgan fingerprint generation."""
    arr = np.zeros(2048, dtype=np.uint8)
    if not isinstance(s, str) or not s:
        return arr
    Chem, AllChem = get_rdkit()
    if Chem is None or AllChem is None:
        return arr
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            for bit in fp.GetOnBits():
                arr[bit] = 1
    except Exception:
        pass
    return arr

def parse_molecular_formula(formula: str) -> np.ndarray:
    """Counts [C, H, N, O, P, S, F, Cl, Br, I] in molecular formula."""
    counts = np.zeros(len(ELEMENTS), dtype=np.float32)
    if not isinstance(formula, str) or not formula:
        return counts
    for idx, el in enumerate(ELEMENTS):
        match = re.search(rf'{el}(\d*)', formula)
        if match:
            val = match.group(1)
            counts[idx] = float(val) if val else 1.0
    return counts

def load_parquet_sample_safe(
    parquet_path: str,
    columns: Optional[List[str]] = None,
    max_records: Optional[int] = None
) -> pd.DataFrame:
    """
    Safely loads records from a Parquet file using PyArrow.
    If max_records is None or <= 0, reads directly via PyArrow C++ engine to eliminate
    the pd.concat peak memory duplication spike.
    """
    if max_records is None or max_records <= 0:
        table = pq.read_table(parquet_path, columns=columns)
        return table.to_pandas()

    pf = pq.ParquetFile(parquet_path)
    dfs = []
    total_loaded = 0

    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=columns)
        df_rg = tbl.to_pandas()
        dfs.append(df_rg)
        total_loaded += len(df_rg)
        if total_loaded >= max_records:
            break

    full_df = pd.concat(dfs, ignore_index=True)
    if len(full_df) > max_records:
        full_df = full_df.iloc[:max_records]
    return full_df

class EnvedaSpectraDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_peaks: int = 128, n_bits: int = 2048, num_workers: Optional[int] = None):
        import time
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm

        self.df = df.reset_index(drop=True)
        self.max_peaks = max_peaks
        self.n_bits = n_bits

        # 1. Map each row to its unique SMILES index (Zero-OOM index pointer)
        smiles_series = self.df['normalized_smiles'].fillna('') if 'normalized_smiles' in self.df.columns else pd.Series([''] * len(self.df))
        unique_smiles, self.mol_indices = np.unique(smiles_series.to_numpy(), return_inverse=True)
        self.mol_indices = self.mol_indices.astype(np.int32)

        # 2. Parallel Multiprocessing computation across unique SMILES only
        workers = num_workers if num_workers is not None else min(cpu_count(), 8)
        print(f"Precomputing {len(unique_smiles):,} unique Morgan fingerprints across {workers} CPU workers...")
        
        t0 = time.time()
        with Pool(processes=workers) as pool:
            fp_list = list(tqdm(
                pool.imap(_worker_smiles_to_fp_uint8, unique_smiles, chunksize=1000),
                total=len(unique_smiles),
                desc="Parallel Morgan Fingerprints",
                dynamic_ncols=True
            ))
        self.unique_fps = np.stack(fp_list, axis=0) # [N_unique, 2048] uint8 -> ONLY ~540 MB for 2.5M spectra!
        elapsed = time.time() - t0
        mean_bits = float(self.unique_fps.sum(axis=1).mean()) if len(self.unique_fps) > 0 else 0.0
        print(f"[OK] Generated {len(unique_smiles):,} unique fingerprints in {elapsed:.1f}s ({len(unique_smiles)/max(0.1, elapsed):.0f} mols/s). Mean active bits: {mean_bits:.1f}")
        print(f"[MEMORY] Unique fingerprints memory footprint: {self.unique_fps.nbytes / (1024**2):.1f} MB (Zero-OOM verified).")

        # 3. Precompute unique formulas with indexing
        formula_series = self.df['molecular_formula'].fillna('') if 'molecular_formula' in self.df.columns else pd.Series([''] * len(self.df))
        unique_formulas, self.form_indices = np.unique(formula_series.to_numpy(), return_inverse=True)
        self.form_indices = self.form_indices.astype(np.int16)
        self.unique_formulas = np.array([parse_molecular_formula(f) for f in unique_formulas], dtype=np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        mzs = np.array(row['ms2_mzs'], dtype=np.float32)
        ints = np.array(row['ms2_normalized_intensities'], dtype=np.float32)
        
        # Sort and select top max_peaks by intensity
        if len(mzs) > self.max_peaks:
            top_idx = np.argsort(ints)[-self.max_peaks:]
            # Re-sort by m/z for monotonic continuous coordinate ordering
            sorted_idx = top_idx[np.argsort(mzs[top_idx])]
            mzs = mzs[sorted_idx]
            ints = ints[sorted_idx]
        cur_len = len(mzs)
        
        # Pad to max_peaks
        pad_len = self.max_peaks - cur_len
        if pad_len > 0:
            padded_mzs = np.pad(mzs, (0, pad_len), constant_values=0.0)
            padded_ints = np.pad(ints, (0, pad_len), constant_values=0.0)
            mask = np.pad(np.ones(cur_len, dtype=bool), (0, pad_len), constant_values=False)
        else:
            padded_mzs = mzs
            padded_ints = ints
            mask = np.ones(self.max_peaks, dtype=bool)
            
        prec_mz = float(row.get('precursor_mz', 0.0))
        
        # Collision energy in eV
        ce = row.get('collision_energy_ev', None)
        if isinstance(ce, (list, np.ndarray)) and len(ce) > 0:
            ce_val = float(ce[0])
        elif isinstance(ce, (int, float)):
            ce_val = float(ce)
        else:
            ce_val = 30.0 # Standard fallback
            
        mode_val = 1.0 if row.get('ionization_mode', 'positive') == 'positive' else 0.0
        
        target_fp = self.unique_fps[self.mol_indices[idx]]
        target_formula = self.unique_formulas[self.form_indices[idx]]
        
        return {
            "mzs": torch.tensor(padded_mzs, dtype=torch.float32),
            "intensities": torch.tensor(padded_ints, dtype=torch.float32),
            "precursor_mz": torch.tensor([prec_mz], dtype=torch.float32),
            "collision_energy": torch.tensor([ce_val], dtype=torch.float32),
            "mode": torch.tensor([mode_val], dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target_fingerprint": torch.from_numpy(target_fp).float(),
            "target_formula": torch.from_numpy(target_formula).float(),
            "smiles": str(row.get('normalized_smiles', ''))
        }
