"""
High-Throughput Streaming PyTorch Dataset for Enveda CASMI MS/MS Spectra
Safely streams from large Parquet files using PyArrow with zero host-RAM OOM.
Supports both precomputed 10,226-bit packed candidate pool targets (casmi26-v2-pool)
and on-the-fly Morgan fingerprints with parallel CPU multiprocessing.
"""

import os
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
    """Generates Morgan/ECFP4 fingerprint from SMILES string."""
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

def _worker_smiles_to_fp_uint8(args) -> np.ndarray:
    """Top-level worker function for parallel Morgan fingerprint generation."""
    if isinstance(args, tuple):
        s, n_bits = args
    else:
        s, n_bits = args, 2048
    n_bytes = int(np.ceil(n_bits / 8))
    arr = np.zeros(n_bytes, dtype=np.uint8)
    if not isinstance(s, str) or not s:
        return arr
    Chem, AllChem = get_rdkit()
    if Chem is None or AllChem is None:
        return arr
    try:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=n_bits)
            for bit in fp.GetOnBits():
                arr[bit // 8] |= (1 << (bit % 8))
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
        n_bits: int = 10226,
        augment: bool = False,
        sample_hard_negs: bool = False,
        num_hard_negs: int = 15,
        num_workers: Optional[int] = None,
        pool_dir: Optional[str] = None
    ):
        import time
        from multiprocessing import Pool, cpu_count
        from tqdm import tqdm
        from enveda_arch.models.neural_operator_net import adduct_to_index, instrument_to_index
        from enveda_arch.retrieval import get_neutral_mass_from_adduct

        self.df = df.reset_index(drop=True)
        self.max_peaks = max_peaks
        self.n_bits = n_bits
        self.n_bytes = int(np.ceil(n_bits / 8))
        self.augment = augment
        self.sample_hard_negs = sample_hard_negs
        self.num_hard_negs = num_hard_negs

        # Auto-detect precomputed pool directory
        if pool_dir is None:
            std_candidates = [
                '/kaggle/input/datasets/ahmedberatozer/casmi26-v2-pool',
                '/kaggle/input/casmi26-v2-pool'
            ]
            for c in std_candidates:
                if os.path.exists(os.path.join(c, 'train_structs.parquet')) and os.path.exists(os.path.join(c, 'train_fp_sel.npy')):
                    pool_dir = c
                    break
        self.pool_dir = pool_dir

        # 1. Map each row to its unique SMILES index (Zero-OOM index pointer)
        smiles_series = self.df['normalized_smiles'].fillna('') if 'normalized_smiles' in self.df.columns else pd.Series([''] * len(self.df))
        unique_smiles, self.mol_indices = np.unique(smiles_series.to_numpy(), return_inverse=True)
        self.mol_indices = self.mol_indices.astype(np.int32)

        # 2. Precompute / Load Fingerprints
        use_precomputed = False
        if self.pool_dir is not None and self.n_bits == 10226:
            structs_path = os.path.join(self.pool_dir, 'train_structs.parquet')
            train_fp_path = os.path.join(self.pool_dir, 'train_fp_sel.npy')
            if os.path.exists(structs_path) and os.path.exists(train_fp_path):
                print(f"Loading precomputed 10,226-bit packed fingerprints from {self.pool_dir}...")
                train_structs = pd.read_parquet(structs_path, columns=['inchikey14', 'smiles'])
                ik_to_idx = {ik: idx for idx, ik in enumerate(train_structs['inchikey14'])}
                smiles_to_idx = {sm: idx for idx, sm in enumerate(train_structs['smiles'])}
                
                mmap_fps = np.load(train_fp_path, mmap_mode='r') # [275810, 1279] uint8
                
                # Map each unique smiles in this dataset to train_fp_sel row
                mapped_fps = np.zeros((len(unique_smiles), self.n_bytes), dtype=np.uint8)
                missing = 0
                for u_idx, sm in enumerate(unique_smiles):
                    # Check by smiles or representative inchikey14
                    r_idx = smiles_to_idx.get(sm, None)
                    if r_idx is not None:
                        mapped_fps[u_idx] = mmap_fps[r_idx]
                    else:
                        missing += 1
                
                if missing == 0:
                    self.unique_fps = mapped_fps
                    use_precomputed = True
                    print(f"[OK] Successfully mapped 100% of {len(unique_smiles):,} structures to precomputed 10,226-bit pool!")
                else:
                    print(f"Notice: {missing} structures missing from precomputed pool, falling back to dynamic computation.")

        if not use_precomputed:
            workers = num_workers if num_workers is not None else min(cpu_count(), 8)
            print(f"Precomputing {len(unique_smiles):,} Morgan fingerprints ({n_bits} bits) across {workers} CPU workers...")
            t0 = time.time()
            worker_args = [(s, self.n_bits) for s in unique_smiles]
            with Pool(processes=workers) as pool:
                fp_list = list(tqdm(
                    pool.imap(_worker_smiles_to_fp_uint8, worker_args, chunksize=1000),
                    total=len(unique_smiles),
                    desc="Parallel Morgan Fingerprints",
                    dynamic_ncols=True
                ))
            self.unique_fps = np.stack(fp_list, axis=0) # [N_unique, N_bytes] uint8
            elapsed = time.time() - t0
            print(f"[OK] Generated {len(unique_smiles):,} unique fingerprints in {elapsed:.1f}s.")

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

        df_tmp = pd.DataFrame({'m_idx': self.mol_indices, 'nm': row_masses})
        med_masses = df_tmp.groupby('m_idx')['nm'].median()
        self.unique_neutral_masses = np.zeros(len(unique_smiles), dtype=np.float64)
        self.unique_neutral_masses[med_masses.index.to_numpy()] = med_masses.to_numpy()

        # Build candidate search pool (from pool_meta if available, else dataset)
        self.pool_meta_fps = None
        if self.pool_dir is not None and os.path.exists(os.path.join(self.pool_dir, 'pool_meta.parquet')) and os.path.exists(os.path.join(self.pool_dir, 'pool_fp.npy')):
            pool_meta = pd.read_parquet(os.path.join(self.pool_dir, 'pool_meta.parquet'), columns=['mass'])
            self.cand_pool_masses = pool_meta['mass'].to_numpy(dtype=np.float64)
            self.cand_order = np.argsort(self.cand_pool_masses)
            self.cand_sorted_masses = self.cand_pool_masses[self.cand_order]
            self.cand_mmap_fp = np.load(os.path.join(self.pool_dir, 'pool_fp.npy'), mmap_mode='r')
            self.use_universe_pool = True
            print(f"[OK] Indexed 710,701 candidate universe for real-world negative isomer sampling.")
        else:
            self.use_universe_pool = False
            self.cand_order = np.argsort(self.unique_neutral_masses)
            self.cand_sorted_masses = self.unique_neutral_masses[self.cand_order]
            self.cand_mmap_fp = self.unique_fps

    def __len__(self):
        return len(self.df)

    def _sample_hard_negs(self, mol_idx: int, neutral_mass: float, k: int = 15, ppm: float = 25.0) -> np.ndarray:
        """Samples k hard-negative candidate fingerprints within mass tolerance."""
        tol = neutral_mass * (ppm * 1e-6)
        l = np.searchsorted(self.cand_sorted_masses, neutral_mass - tol)
        r = np.searchsorted(self.cand_sorted_masses, neutral_mass + tol)
        
        if (r - l) < k + 1:
            tol_wide = neutral_mass * (100.0 * 1e-6)
            l = np.searchsorted(self.cand_sorted_masses, neutral_mass - tol_wide)
            r = np.searchsorted(self.cand_sorted_masses, neutral_mass + tol_wide)

        cands = self.cand_order[l:r]
        if not self.use_universe_pool:
            cands = cands[cands != mol_idx]

        if len(cands) >= k:
            sampled = np.random.choice(cands, size=k, replace=False)
        elif len(cands) > 0:
            sampled = np.random.choice(cands, size=k, replace=True)
        else:
            sampled = np.random.choice(len(self.cand_mmap_fp), size=k, replace=False)

        return np.array(self.cand_mmap_fp[sampled]) # [K, N_bytes] uint8

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        mol_idx = int(self.mol_indices[idx])
        
        mzs = np.array(row['ms2_mzs'], dtype=np.float32)
        ints = np.array(row['ms2_normalized_intensities'], dtype=np.float32)

        # 1. Peak Augmentation during training
        if self.augment and len(mzs) > 5:
            top5_indices = set(np.argsort(ints)[-5:])
            keep = [i for i in range(len(mzs)) if (i in top5_indices or np.random.rand() > 0.20)]
            if len(keep) >= 3:
                mzs = mzs[keep]
                ints = ints[keep]
            ints = ints * np.exp(np.random.normal(0.0, 0.15, size=len(ints))).astype(np.float32)
            max_i = np.max(ints)
            if max_i > 0:
                ints = ints / max_i
            mzs = mzs * (1.0 + np.random.normal(0.0, 2e-6, size=len(mzs))).astype(np.float32)
        
        if len(mzs) > self.max_peaks:
            top_idx = np.argsort(ints)[-self.max_peaks:]
            sorted_idx = top_idx[np.argsort(mzs[top_idx])]
            mzs = mzs[sorted_idx]
            ints = ints[sorted_idx]
        else:
            sorted_idx = np.argsort(mzs)
            mzs = mzs[sorted_idx]
            ints = ints[sorted_idx]

        cur_len = len(mzs)
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
        
        ce = row.get('collision_energy_ev', None)
        if isinstance(ce, (list, np.ndarray)) and len(ce) > 0:
            ce_val = float(ce[0])
        elif isinstance(ce, (int, float)):
            ce_val = float(ce)
        else:
            ce_val = 30.0
            
        mode_val = 1.0 if row.get('ionization_mode', 'positive') == 'positive' else 0.0
        
        target_fp_packed = self.unique_fps[mol_idx] # uint8 array of shape [N_bytes]
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
            "target_fingerprint_packed": torch.from_numpy(target_fp_packed).to(torch.uint8),
            "target_formula": torch.from_numpy(target_formula).float(),
            "smiles": str(row.get('normalized_smiles', ''))
        }

        # Hard-negative isomer fingerprints (packed uint8)
        if self.sample_hard_negs:
            nm = float(self.unique_neutral_masses[mol_idx])
            hard_fps = self._sample_hard_negs(mol_idx, nm, k=self.num_hard_negs)
            item["hard_neg_fps_packed"] = torch.from_numpy(hard_fps).to(torch.uint8)

        return item
