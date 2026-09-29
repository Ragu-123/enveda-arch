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

    clean_peaks = _clean_numba
    entropy_similarity = _entropy_sim_numba
else:
    clean_peaks = _clean_numpy
    entropy_similarity = _entropy_sim_numpy
