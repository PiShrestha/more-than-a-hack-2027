"""Fixed, label-independent ball-observation stress cases. Player noise stays paired."""
import numpy as np

CASES=('visible','target_like','iid40','burst2s40','burst10s40')
SEEDS=(101,202,303)

def mask(case,n,template,seed):
    if case not in CASES or n<1:
        raise ValueError('unknown case or invalid length')
    rng=np.random.default_rng(np.random.SeedSequence([seed,CASES.index(case)]))
    if case=='visible':
        return np.zeros(n,dtype=bool)
    if case=='target_like':
        template=np.asarray(template,dtype=bool)
        if not len(template):
            raise ValueError('empty target observation template')
        start=int(rng.integers(len(template)))
        return np.resize(template,n+start)[start:].copy()
    if case=='iid40':
        return rng.random(n)<0.4
    mean_gap=50 if case=='burst2s40' else 250
    p_on=1/mean_gap
    p_off=0.4*p_on/0.6
    state=bool(rng.random()<0.4)  # stationary initial distribution
    out=np.empty(n,dtype=bool)
    for i,u in enumerate(rng.random(n)):
        state=(u>=p_on) if state else (u<p_off)
        out[i]=state
    return out

def gap_summary(missing):
    a=np.asarray(missing,dtype=bool)
    diff=np.diff(np.concatenate(([False],a,[False])).astype(np.int8))
    gaps=np.flatnonzero(diff==-1)-np.flatnonzero(diff==1)
    return dict(missing_fraction=float(a.mean()),gaps=int(len(gaps)),
                mean_gap_s=float(gaps.mean()/25) if len(gaps) else 0.,
                max_gap_s=float(gaps.max()/25) if len(gaps) else 0.)
