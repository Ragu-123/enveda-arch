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

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem
    HAS_RDKIT = True
except ImportError:
    HAS_RDKIT = False

ELEMENTS = ['C', 'H', 'N', 'O', 'P', 'S', 'F', 'Cl', 'Br', 'I']

def smiles_to_morgan_fingerprint(smiles: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Generates 2048-bit Morgan/ECFP4 fingerprint from SMILES string."""
    fp_arr = np.zeros((n_bits,), dtype=np.float32)
    if not HAS_RDKIT or not isinstance(smiles, str) or not smiles:
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
        
        # Ground truth targets
        smiles = row.get('normalized_smiles', '')
        fp = smiles_to_morgan_fingerprint(smiles, n_bits=self.n_bits)
        formula = row.get('molecular_formula', '')
        form_counts = parse_molecular_formula(formula)
        
        return {
            "mzs": torch.tensor(padded_mzs, dtype=torch.float32),
            "intensities": torch.tensor(padded_ints, dtype=torch.float32),
            "precursor_mz": torch.tensor([prec_mz], dtype=torch.float32),
            "collision_energy": torch.tensor([ce_val], dtype=torch.float32),
            "mode": torch.tensor([mode_val], dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "target_fingerprint": torch.tensor(fp, dtype=torch.float32),
            "target_formula": torch.tensor(form_counts, dtype=torch.float32)
        }
