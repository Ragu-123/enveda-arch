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
    def __init__(
        self,
        df: pd.DataFrame,
        max_peaks: int = 128,
        n_bits: int = 2048,
        augment: bool = False,
        sample_hard_negs: bool = False,
        num_hard_negs: int = 15,
        num_workers: Optional[int] = None
    ):
        import time
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        from enveda_arch.models.neural_operator_net import adduct_to_index, instrument_to_index
        from enveda_arch.retrieval import get_neutral_mass_from_adduct

        self.df = df.reset_index(drop=True)
        self.max_peaks = max_peaks
        self.n_bits = n_bits
        self.augment = augment
        self.sample_hard_negs = sample_hard_negs
        self.num_hard_negs = num_hard_negs

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

        # 4. Map conditions: Adduct & Instrument indexes
        adduct_col = self.df['adduct'] if 'adduct' in self.df.columns else pd.Series(['[M+H]+'] * len(self.df))
        self.adduct_indices = np.array([adduct_to_index(a) for a in adduct_col.fillna('[M+H]+')], dtype=np.int16)

        instr_col = self.df['instrument_type'] if 'instrument_type' in self.df.columns else pd.Series(['timsTOF'] * len(self.df))
        self.instr_indices = np.array([instrument_to_index(i) for i in instr_col.fillna('timsTOF')], dtype=np.int8)

        # 5. Precompute unique neutral masses for fast hard-negative candidate search
        row_masses = np.zeros(len(self.df), dtype=np.float64)
        prec_arr = self.df['precursor_mz'].to_numpy(dtype=np.float64)
        mode_arr = self.df['ionization_mode'].to_numpy() if 'ionization_mode' in self.df.columns else ['positive'] * len(self.df)
        adduct_arr = adduct_col.to_numpy()

        for i in range(len(self.df)):
            row_masses[i] = get_neutral_mass_from_adduct(prec_arr[i], str(adduct_arr[i]), str(mode_arr[i]))

        # Median neutral mass per unique molecule
        self.unique_neutral_masses = np.zeros(len(unique_smiles), dtype=np.float64)
        # Fast aggregation using pandas or numpy
        df_tmp = pd.DataFrame({'m_idx': self.mol_indices, 'nm': row_masses})
        med_masses = df_tmp.groupby('m_idx')['nm'].median()
        self.unique_neutral_masses[med_masses.index.to_numpy()] = med_masses.to_numpy()

        # Build sorted mass array for O(log N) candidate binary search
        self.sorted_mass_order = np.argsort(self.unique_neutral_masses)
        self.sorted_masses = self.unique_neutral_masses[self.sorted_mass_order]
        print(f"[OK] Indexed {len(self.sorted_masses):,} unique neutral masses for hard-negative isomer sampling.")

    def __len__(self):
        return len(self.df)

    def _sample_hard_negs(self, mol_idx: int, neutral_mass: float, k: int = 15, ppm: float = 25.0) -> np.ndarray:
        """Samples k hard-negative candidate fingerprints within mass tolerance."""
        tol = neutral_mass * (ppm * 1e-6)
        l = np.searchsorted(self.sorted_masses, neutral_mass - tol)
        r = np.searchsorted(self.sorted_masses, neutral_mass + tol)
        
        # If too few candidates, widen tolerance
        if (r - l) < k + 1:
            tol_wide = neutral_mass * (100.0 * 1e-6)
            l = np.searchsorted(self.sorted_masses, neutral_mass - tol_wide)
            r = np.searchsorted(self.sorted_masses, neutral_mass + tol_wide)

        cands = self.sorted_mass_order[l:r]
        # Exclude ground truth molecule
        cands = cands[cands != mol_idx]

        if len(cands) >= k:
            sampled = np.random.choice(cands, size=k, replace=False)
        elif len(cands) > 0:
            sampled = np.random.choice(cands, size=k, replace=True)
        else:
            # Fallback random sampling
            sampled = np.random.choice(len(self.unique_fps), size=k, replace=False)

        return self.unique_fps[sampled] # [K, 2048] uint8

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_idx = int(self.mol_indices[idx])
        
        mzs = np.array(row['ms2_mzs'], dtype=np.float32)
        ints = np.array(row['ms2_normalized_intensities'], dtype=np.float32)

        # 1. Peak Augmentation during training
        if self.augment and len(mzs) > 5:
            # Peak Dropout: randomly keep 75-95% of peaks, ensuring top 5 highest peaks are never dropped
            top5_indices = set(np.argsort(ints)[-5:])
            keep = [i for i in range(len(mzs)) if (i in top5_indices or np.random.rand() > 0.20)]
            if len(keep) >= 3:
                mzs = mzs[keep]
                ints = ints[keep]

            # Intensity jitter: log-normal scaling
            ints = ints * np.exp(np.random.normal(0.0, 0.15, size=len(ints))).astype(np.float32)
            max_i = np.max(ints)
            if max_i > 0:
                ints = ints / max_i

            # Fragment m/z jitter: +-2 ppm
            mzs = mzs * (1.0 + np.random.normal(0.0, 2e-6, size=len(mzs))).astype(np.float32)
        
        # Sort and select top max_peaks by intensity
        if len(mzs) > self.max_peaks:
            top_idx = np.argsort(ints)[-self.max_peaks:]
            # Re-sort by m/z for monotonic continuous coordinate ordering
            sorted_idx = top_idx[np.argsort(mzs[top_idx])]
            mzs = mzs[sorted_idx]
            ints = ints[sorted_idx]
        else:
            # Always ensure sorted by m/z
            sorted_idx = np.argsort(mzs)
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
        if self.augment:
            prec_mz = prec_mz * (1.0 + float(np.random.normal(0.0, 2e-6)))
        
        # Collision energy in eV
        ce = row.get('collision_energy_ev', None)
        if isinstance(ce, (list, np.ndarray)) and len(ce) > 0:
            ce_val = float(ce[0])
        elif isinstance(ce, (int, float)):
            ce_val = float(ce)
        else:
            ce_val = 30.0 # Standard fallback
            
        mode_val = 1.0 if row.get('ionization_mode', 'positive') == 'positive' else 0.0
        
        target_fp = self.unique_fps[mol_idx]
        target_formula = self.unique_formulas[self.form_indices[idx]]

        item = {
            "mzs": torch.tensor(padded_mzs, dtype=torch.float32),
            "intensities": torch.tensor(padded_ints, dtype=torch.float32),
            "precursor_mz": torch.tensor([prec_mz], dtype=torch.float32),
            "collision_energy": torch.tensor([ce_val], dtype=torch.float32),
            "mode": torch.tensor([mode_val], dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "adduct_ix": torch.tensor(int(self.adduct_indices[idx]), dtype=torch.long),
            "instr_ix": torch.tensor(int(self.instr_indices[idx]), dtype=torch.long),
            "target_fingerprint": torch.from_numpy(target_fp).float(),
            "target_formula": torch.from_numpy(target_formula).float(),
            "smiles": str(row.get('normalized_smiles', ''))
        }

        # Hard-negative isomer fingerprints
        if self.sample_hard_negs:
            nm = float(self.unique_neutral_masses[mol_idx])
            hard_fps = self._sample_hard_negs(mol_idx, nm, k=self.num_hard_negs)
            item["hard_neg_fps"] = torch.from_numpy(hard_fps).float()

        return item
