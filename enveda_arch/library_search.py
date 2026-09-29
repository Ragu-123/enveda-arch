"""
Spectral Entropy Similarity & Direct Library Search Engine
Implements fast continuous spectral entropy (Li et al. Nature Methods 2021) and
precursor-shifted matching to detect exact and analog library references.
"""

import numpy as np
try:
    from numba import njit, prange
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False

def _clean_numpy(mz, it, floor=0.002, topk=256, power=1.0, ent_weight=True):
    if len(mz) == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = float(it.max())
    if mx <= 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    mask = it >= thr
    if not mask.any():
        return np.empty(0, np.float32), np.empty(0, np.float32)
    idx = np.where(mask)[0]
    if len(idx) > topk:
        top_idx = np.argsort(it[idx])[-topk:]
        idx = np.sort(idx[top_idx])
    om = mz[idx].astype(np.float32)
    oi = (it[idx] ** power).astype(np.float32)
    s = oi.sum()
    if s > 0:
        oi /= s
    if ent_weight:
        pos = oi > 0
        S = -np.sum(oi[pos] * np.log(oi[pos]))
        if S < 3.0:
            w = 0.25 + 0.25 * S
            oi = oi ** w
            s2 = oi.sum()
            if s2 > 0:
                oi /= s2
    return om, oi

def _entropy_sim_numpy(qmz, qp, cmz, cp, tol=0.015):
    if len(qmz) == 0 or len(cmz) == 0:
        return 0.0
    pos_q = qp > 0
    SA = -float(np.sum(qp[pos_q] * np.log(qp[pos_q])))
    pos_c = cp > 0
    SB = -float(np.sum(cp[pos_c] * np.log(cp[pos_c])))
    
    # Merge peaks within tolerance
    i = 0
    j = 0
    n = len(qmz)
    m = len(cmz)
    buf = []
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf.append(qp[i])
            i += 1
        elif d > tol:
            buf.append(cp[j])
            j += 1
        else:
            buf.append(qp[i] + cp[j])
            i += 1
            j += 1
    while i < n:
        buf.append(qp[i])
        i += 1
    while j < m:
        buf.append(cp[j])
        j += 1
        
    buf = np.array(buf, dtype=np.float64)
    tot = buf.sum()
    if tot <= 0:
        return 0.0
    v = buf / tot
    pos_v = v > 0
    SAB = -float(np.sum(v[pos_v] * np.log(v[pos_v])))
    sim = 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)
    return max(0.0, min(1.0, float(sim)))

if HAS_NUMBA:
    @njit(cache=True, fastmath=True)
    def _clean_numba(mz, it, floor, topk, power, ent_weight):
        n = len(mz)
        if n == 0:
            return np.empty(0, np.float32), np.empty(0, np.float32)
        mx = 0.0
        for i in range(n):
            if it[i] > mx:
                mx = it[i]
        if mx <= 0:
            return np.empty(0, np.float32), np.empty(0, np.float32)
        thr = floor * mx
        c = 0
        for i in range(n):
            if it[i] >= thr:
                c += 1
        idx = np.empty(c, np.int64)
        j = 0
        for i in range(n):
            if it[i] >= thr:
                idx[j] = i
                j += 1
        if c > topk:
            v = np.empty(c, np.float32)
            for i in range(c):
                v[i] = it[idx[i]]
            o = np.argsort(v)[c - topk:]
            k2 = np.empty(topk, np.int64)
            for i in range(topk):
                k2[i] = idx[o[i]]
            k2.sort()
            idx = k2
            c = topk
        om = np.empty(c, np.float32)
        oi = np.empty(c, np.float32)
        s = 0.0
        for i in range(c):
            om[i] = mz[idx[i]]
            v = it[idx[i]] ** power
            oi[i] = v
            s += v
        if s > 0:
            for i in range(c):
                oi[i] /= s
        if ent_weight:
            S = 0.0
            for i in range(c):
                if oi[i] > 0:
                    S -= oi[i] * np.log(oi[i])
            if S < 3.0:
                w = 0.25 + 0.25 * S
                s2 = 0.0
                for i in range(c):
                    oi[i] = oi[i] ** w
                    s2 += oi[i]
                if s2 > 0:
                    for i in range(c):
                        oi[i] /= s2
        return om, oi

    @njit(cache=True, fastmath=True)
    def _entropy_sim_numba(qmz, qp, cmz, cp, tol):
        i = 0
        j = 0
        n = len(qmz)
        m = len(cmz)
        SA = 0.0
        for x in range(n):
            if qp[x] > 0:
                SA -= qp[x] * np.log(qp[x])
        SB = 0.0
        for x in range(m):
            if cp[x] > 0:
                SB -= cp[x] * np.log(cp[x])
        SAB = 0.0
        tot = 0.0
        buf = np.empty(n + m, np.float64)
        b = 0
        while i < n and j < m:
            d = qmz[i] - cmz[j]
            if d < -tol:
                buf[b] = qp[i]
                i += 1
                b += 1
            elif d > tol:
                buf[b] = cp[j]
                j += 1
                b += 1
            else:
                buf[b] = qp[i] + cp[j]
                i += 1
                j += 1
                b += 1
        while i < n:
            buf[b] = qp[i]
            i += 1
            b += 1
        while j < m:
            buf[b] = cp[j]
            j += 1
            b += 1
        for x in range(b):
            tot += buf[x]
        if tot <= 0:
            return 0.0
        for x in range(b):
            v = buf[x] / tot
            if v > 0:
                SAB -= v * np.log(v)
        return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

def clean_peaks(mz: np.ndarray, it: np.ndarray, floor: float = 0.005, topk: int = 48, power: float = 1.0, ent_weight: bool = True):
    if HAS_NUMBA:
        return _clean_numba(np.asarray(mz, np.float32), np.asarray(it, np.float32), float(floor), int(topk), float(power), bool(ent_weight))
    return _clean_numpy(np.asarray(mz, np.float32), np.asarray(it, np.float32), float(floor), int(topk), float(power), bool(ent_weight))

def entropy_similarity(qmz: np.ndarray, qp: np.ndarray, cmz: np.ndarray, cp: np.ndarray, tol: float = 0.015) -> float:
    if HAS_NUMBA:
        return float(_entropy_sim_numba(np.asarray(qmz, np.float32), np.asarray(qp, np.float32), np.asarray(cmz, np.float32), np.asarray(cp, np.float32), float(tol)))
    return float(_entropy_sim_numpy(np.asarray(qmz, np.float32), np.asarray(qp, np.float32), np.asarray(cmz, np.float32), np.asarray(cp, np.float32), float(tol)))


class MassBankLibrary:
    """
    In-memory indexed reference library for fast precursor-windowed and neutral-mass-windowed
    spectral entropy search. Targets Tier 1 exact reference matching in CASMI 2026.
    Supports dual-index search:
      1. Precursor m/z window (+-10 ppm) for same-adduct matching
      2. Exact neutral mass window (+-10 ppm) for cross-adduct matching
    """
    def __init__(self, parquet_path: str = "/kaggle/input/datasets/samartalwar/casmi-2026-spectral-library-massbankharmonized/spectra.parquet"):
        self.parquet_path = parquet_path
        self.loaded = False
        self.precursor_mzs = np.empty(0, np.float32)
        self.exact_masses = np.empty(0, np.float32)
        self.sorted_exact_masses = np.empty(0, np.float32)
        self.exact_mass_order = np.empty(0, np.int64)
        self.smiles = []
        self.inchikeys = []
        self.inchikey14s = []
        self.ion_modes = []
        self.mzs_list = []
        self.ints_list = []

    def load(self, min_mz: float = 230.0, max_mz: float = 475.0) -> bool:
        import os
        if not os.path.exists(self.parquet_path):
            print(f"[WARN] MassBank parquet not found at: {self.parquet_path}")
            return False
        import pyarrow.parquet as pq
        cols = ['smiles', 'inchikey', 'precursor_mz', 'exact_mass', 'ion_mode', 'mzs', 'intensities']
        tbl = pq.read_table(self.parquet_path, columns=cols)
        df = tbl.to_pandas()
        
        # Filter to competition window (either precursor or exact mass within window)
        mask = (
            ((df['precursor_mz'] >= min_mz) & (df['precursor_mz'] <= max_mz)) |
            ((df['exact_mass'] >= min_mz - 5.0) & (df['exact_mass'] <= max_mz + 5.0))
        )
        df_sub = df[mask].reset_index(drop=True)
        if len(df_sub) == 0:
            df_sub = df
        
        # Sort primarily by precursor_mz for fast search
        df_sub = df_sub.sort_values('precursor_mz').reset_index(drop=True)
        self.precursor_mzs = df_sub['precursor_mz'].to_numpy(dtype=np.float32)
        self.exact_masses = df_sub['exact_mass'].to_numpy(dtype=np.float32)
        
        # Secondary index: sorted order by exact_mass
        self.exact_mass_order = np.argsort(self.exact_masses)
        self.sorted_exact_masses = self.exact_masses[self.exact_mass_order]
        
        self.smiles = df_sub['smiles'].tolist()
        self.inchikeys = df_sub['inchikey'].tolist()
        self.inchikey14s = [str(k)[:14] if isinstance(k, str) else '' for k in self.inchikeys]
        self.ion_modes = df_sub['ion_mode'].astype(str).str.upper().tolist()
        self.mzs_list = [np.array(m, dtype=np.float32) for m in df_sub['mzs']]
        self.ints_list = [np.array(it, dtype=np.float32) for it in df_sub['intensities']]
        self.loaded = True
        print(f"[OK] MassBankLibrary loaded {len(self.precursor_mzs):,} reference spectra in [{min_mz}, {max_mz}] Da (Dual Precursor & Neutral Mass Indexes active).")
        return True

    def query(self, prec_mz: float, neutral_mass: Optional[float], mode_str: str, qmz: np.ndarray, qit: np.ndarray,
              tol_ppm: float = 12.0, min_sim: float = 0.85, min_peaks: int = 4):
        """
        Query reference library using both precursor m/z and deconvoluted neutral mass.
        FDR Guardrails:
          - Requires at least `min_peaks` (default: 4) clean fragment peaks
          - Strict spectral entropy threshold `min_sim` (default: 0.85)
          - Precursor / neutral mass tolerance `tol_ppm` (default: 12.0 ppm)
        """
        if not self.loaded:
            return []

        # 1. FDR Guardrail: require sufficient query peak information
        qm_c, qi_c = clean_peaks(qmz, qit, 0.005, 48, 1.0, True)
        if len(qm_c) < min_peaks:
            return []

        target_mode = "POSITIVE" if ("POS" in str(mode_str).upper() or mode_str in ("1", "1.0", 1)) else "NEGATIVE"
        
        candidate_indices = set()
        
        # 2. Precursor m/z window search (+- tol_ppm)
        tol_da = prec_mz * (tol_ppm * 1e-6)
        idx_l = int(np.searchsorted(self.precursor_mzs, prec_mz - tol_da))
        idx_r = int(np.searchsorted(self.precursor_mzs, prec_mz + tol_da))
        for i in range(idx_l, idx_r):
            if self.ion_modes[i] == target_mode:
                candidate_indices.add(i)

        # 3. Cross-adduct neutral mass window search (+- tol_ppm)
        if neutral_mass is not None and neutral_mass > 50.0:
            tol_neutral = neutral_mass * (tol_ppm * 1e-6)
            n_idx_l = int(np.searchsorted(self.sorted_exact_masses, neutral_mass - tol_neutral))
            n_idx_r = int(np.searchsorted(self.sorted_exact_masses, neutral_mass + tol_neutral))
            for k in range(n_idx_l, n_idx_r):
                orig_i = int(self.exact_mass_order[k])
                if self.ion_modes[orig_i] == target_mode:
                    candidate_indices.add(orig_i)

        if not candidate_indices:
            return []

        # 4. Evaluate spectral entropy similarity
        matches = []
        for i in candidate_indices:
            cm_c, ci_c = clean_peaks(self.mzs_list[i], self.ints_list[i], 0.005, 48, 1.0, True)
            if len(cm_c) < 3:
                continue
            sim = entropy_similarity(qm_c, qi_c, cm_c, ci_c, 0.015)
            if sim >= min_sim:
                matches.append({
                    'smiles': self.smiles[i],
                    'inchikey': self.inchikeys[i],
                    'inchikey14': self.inchikey14s[i],
                    'similarity': float(sim),
                    'library_prec_mz': float(self.precursor_mzs[i]),
                    'library_exact_mass': float(self.exact_masses[i])
                })
        
        # Rank by spectral entropy similarity descending
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        return matches
