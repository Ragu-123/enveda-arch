"""
In-Silico Fragmentation & Explainability Engine (MetFrag-lite with Adduct Charge Carriers)
Simulates 1- and 2-bond neutral cleavages on candidate molecular graphs to break
the constitutional isomer bottleneck (98% of candidate ranking failures).
"""

import numpy as np
from typing import List, Tuple, Optional, Set
try:
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    HAS_RDKIT = True
except Exception:
    Chem = None
    HAS_RDKIT = False

# Monoisotopic atomic weights
AMU = {
    'C': 12.0,
    'H': 1.00782503207,
    'N': 14.0030740048,
    'O': 15.9949146196,
    'P': 30.97376163,
    'S': 31.97207100,
    'F': 18.99840322,
    'Cl': 34.96885268,
    'Br': 78.9183371,
    'I': 126.904473,
    'Na': 22.9897692809,
    'K': 38.96370668,
    'Si': 27.9769265325,
    'B': 11.0093054,
    'Se': 79.9165213
}
E_MASS = 0.00054857990
H_MASS = AMU['H']
PROTON = H_MASS - E_MASS
H2O = 2 * H_MASS + AMU['O']
NH4 = AMU['N'] + 4 * H_MASS
FORMATE = AMU['C'] + 2 * H_MASS + 2 * AMU['O']
ACETATE = 2 * AMU['C'] + 4 * H_MASS + 2 * AMU['O']

CARRIERS = {
    "[M+H]+": (PROTON,),
    "[M-H2O+H]+": (PROTON,),
    "[M-2H2O+H]+": (PROTON,),
    "[M+NH4]+": (PROTON, NH4 - E_MASS),
    "[M+Na]+": (PROTON, AMU['Na'] - E_MASS),
    "[M+K]+": (PROTON, AMU['K'] - E_MASS),
    "[M-H]-": (-PROTON,),
    "[M-H2O-H]-": (-PROTON,),
    "[M+CH2O2-H]-": (-PROTON, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (-PROTON, ACETATE - PROTON),
    "[M+Cl]-": (-PROTON, AMU['Cl'] + E_MASS),
}

def mol_graph(smi: str):
    if not HAS_RDKIT or Chem is None:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    n = m.GetNumAtoms()
    w = np.zeros(n, dtype=np.float64)
    for a in m.GetAtoms():
        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H_MASS
    if (w == 0).any():
        return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n

def _components(n: int, bonds: list, drop: Set[int]) -> List[List[int]]:
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop:
            continue
        adj[a].append(b)
        adj[b].append(a)
    seen = np.zeros(n, dtype=bool)
    comps = []
    for s in range(n):
        if seen[s]:
            continue
        stack = [s]
        seen[s] = True
        cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]:
                    seen[v] = True
                    stack.append(v)
                    cur.append(v)
        comps.append(cur)
    return comps

def fragment_masses(smi: str, max_breaks: int = 2, max_bonds: int = 34) -> np.ndarray:
    """
    Simulates 1- and 2-bond neutral cleavages on candidate molecular graph.
    Returns sorted array of unique theoretical fragment masses.
    """
    g = mol_graph(smi)
    if g is None:
        return np.zeros(0, dtype=np.float64)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds:
        return np.array([w.sum()], dtype=np.float64)
    
    out = {w.sum()}
    # 1-bond breaks
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
            
    # 2-bond breaks
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
                    
    return np.array(sorted(out), dtype=np.float64)

def frag_masses_safe(smi: str) -> np.ndarray:
    try:
        return fragment_masses(smi)
    except Exception:
        return np.zeros(0, dtype=np.float64)

def explain_score_adduct(
    frag_mass: np.ndarray,
    peak_mz: np.ndarray,
    peak_int: np.ndarray,
    adduct: str,
    tol: float = 0.015,
    h_shifts: Tuple[int, ...] = (-2, -1, 0, 1, 2)
) -> float:
    """
    Quantifies the intensity-weighted fraction of observed MS/MS peaks accounted for
    by theoretical fragments, modeling adduct-specific charge carriers and radical shifts.
    """
    if len(frag_mass) == 0 or len(peak_mz) == 0:
        return 0.0
    car = CARRIERS.get(adduct, (PROTON,) if (isinstance(adduct, str) and "+" in adduct) else (-PROTON,))
    ion = np.sort(
        np.concatenate([frag_mass + dh * H_MASS + c for dh in h_shifts for c in car])
    )
    w = np.sqrt(np.asarray(peak_int, dtype=np.float64))
    tot = w.sum()
    if tot <= 0:
        return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), dtype=bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)
