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
    max_records: int = 25000
) -> pd.DataFrame:
    """
    Safely loads a slice from a large Parquet file using PyArrow row groups,
    preventing entire-file host RAM exhaustion.
    """
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
    def __init__(self, df: pd.DataFrame, max_peaks: int = 128, n_bits: int = 2048):
        self.df = df.reset_index(drop=True)
        self.max_peaks = max_peaks
        self.n_bits = n_bits

        # Precompute fingerprints for all samples
        print(f"Precomputing {len(self.df)} ground-truth Morgan fingerprints...")
        smiles_list = self.df['normalized_smiles'].tolist() if 'normalized_smiles' in self.df.columns else []
        fps_list = []
        Chem, AllChem = get_rdkit()
        if Chem is None:
            raise RuntimeError("RDKit is NOT available! Install rdkit via pip before building dataset.")
            
        for s in smiles_list:
            fps_list.append(smiles_to_morgan_fingerprint(s, n_bits=n_bits))
        self.fps = np.stack(fps_list, axis=0) if len(fps_list) > 0 else np.zeros((len(self.df), n_bits), dtype=np.float32)
        
        mean_bits = float(self.fps.sum(axis=1).mean()) if len(self.fps) > 0 else 0.0
        print(f"[OK] Precomputed fingerprints. Mean active bits per molecule: {mean_bits:.1f}")
        if len(self.fps) > 0 and mean_bits < 1.0:
            raise RuntimeError(f"CRITICAL ERROR: Mean active bits is {mean_bits:.2f}! Fingerprint generation failed!")

        # Precompute formula counts
        formulas = self.df['molecular_formula'].tolist() if 'molecular_formula' in self.df.columns else []
        self.formulas = np.array([parse_molecular_formula(f) for f in formulas], dtype=np.float32) if len(formulas) > 0 else np.zeros((len(self.df), len(ELEMENTS)), dtype=np.float32)

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
        
        target_fp = self.fps[idx]
        target_formula = self.formulas[idx]
        
        return {
            "mzs": torch.tensor(padded_mzs, dtype=torch.float32),
            "intensities": torch.tensor(padded_ints, dtype=torch.float32),
            "precursor_mz": torch.tensor([prec_mz], dtype=torch.float32),
            "collision_energy": torch.tensor([ce_val], dtype=torch.float32),
            "mode": torch.tensor([mode_val], dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target_fingerprint": torch.tensor(target_fp, dtype=torch.float32),
            "target_formula": torch.tensor(target_formula, dtype=torch.float32)
        }
