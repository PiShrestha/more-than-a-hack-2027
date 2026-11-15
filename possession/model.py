"""Train a per-frame H/A/D classifier on IDSSE (other games), decode with Viterbi, predict BAR-ATH.

    python -m possession.model cv        # leave-matches-out validation on IDSSE, tunes the decoder
    python -m possession.model predict   # train on all IDSSE, write the BAR-ATH submission

No label from the provided game is used anywhere. BAR-ATH contributes only its *ball-missing
mask* (an observation pattern, not a label) to make the IDSSE training data look like its noise.
"""

import json
import os
import sys

import numpy as np
from numba import njit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import idsse  # noqa: E402
from possession.data import bar_ath  # noqa: E402
from possession.features import features  # noqa: E402

OUT = os.path.join(ROOT, "submission")
DECODER = os.path.join(ROOT, "possession", "decoder.json")
SEED = 0
LGB = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
           lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=8,
           deterministic=True, force_col_wise=True)
N_ROUNDS = 400
SUBSAMPLE = 3  # train on every 3rd frame: neighbouring frames are near-duplicates


# -- noise model --------------------------------------------------------------------------------

def bar_ath_ball_mask():
    d = bar_ath()
    return np.isnan(d["bx"][d["period"] > 0])


def add_noise(d, ball_mask, rng):
    """Make clean IDSSE tracking look like the automatic ATD.

    ball: BAR-ATH's own graded ball-missing mask, cut at a random offset, so gap lengths and their
    drift through a half are reproduced exactly. players: each one drops out as a two-state Markov
    chain (~7% missing, mean outage ~2 s, matching 20.5 of 22 tracked), plus 0.15 m jitter.
    """
    d = dict(d)
    T, S = d["px"].shape
    reps = int(np.ceil(T / len(ball_mask))) + 1
    m = np.tile(ball_mask, reps)
    off = rng.integers(len(ball_mask))
    bm = m[off:off + T]
    d["bx"] = np.where(bm, np.nan, d["bx"]).astype(np.float32)
    d["by"] = np.where(bm, np.nan, d["by"]).astype(np.float32)
    p_off, p_on = 0.07 / 50 / 0.93, 1 / 50
    state = np.zeros(S, bool)
    drop = np.zeros((T, S), bool)
    u = rng.random((T, S), dtype=np.float32)
    for t in range(T):
        state = np.where(state, u[t] > p_on, u[t] < p_off)
        drop[t] = state
    jit = rng.normal(0, 0.15, (2, T, S)).astype(np.float32)
    d["px"] = np.where(drop, np.nan, d["px"] + jit[0])
    d["py"] = np.where(drop, np.nan, d["py"] + jit[1])
    return d


def dataset(matches, ball_mask, seed=SEED, mirror=True, sub=SUBSAMPLE):
    """Stacked features/labels over matches, noisy, optionally mirrored (teams swapped)."""
    Xs, ys = [], []
    rng = np.random.default_rng(seed)
    for m in matches:
        d = add_noise(idsse.load(m), ball_mask, rng)
        X, names = features(d)
        y = d["y"].astype(np.int64)
        idx = np.flatnonzero(y >= 0)[::sub]
        Xs.append(X[idx])
        ys.append(y[idx])
        if mirror:
            Xf, _ = features(d, flip=True)
            yf = np.where(y == 0, 1, np.where(y == 1, 0, y))
            Xs.append(Xf[idx])
            ys.append(yf[idx])
    return np.concatenate(Xs), np.concatenate(ys), names


def train(X, y):
    import lightgbm as lgb
    return lgb.train(LGB, lgb.Dataset(X, y), N_ROUNDS)


def predict_proba(model, d):
    """Symmetrised: average p(H,A,D | x) with the team-swapped view."""
    X, _ = features(d)
    Xf, _ = features(d, flip=True)
    p = model.predict(X)
    pf = model.predict(Xf)[:, [1, 0, 2]]
    return 0.5 * (p + pf)


# -- decoding -----------------------------------------------------------------------------------

@njit(cache=True)
def _viterbi(logp, trans):
    T = logp.shape[0]
    back = np.zeros((T, 3), np.int8)
    score = logp[0].copy()
    new = np.empty(3)
    for t in range(1, T):
        for j in range(3):
            best, arg = -1e18, 0
            for i in range(3):
                v = score[i] + trans[i, j]
                if v > best:
                    best, arg = v, i
            new[j] = best + logp[t, j]
            back[t, j] = arg
        score[:] = new
    path = np.zeros(T, np.int8)
    path[-1] = np.argmax(score)
    for t in range(T - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def viterbi(logp, switch, die, revive):
    """Max-score path over states H, A, D with per-transition penalties (log domain)."""
    trans = -np.array([[0, switch, die], [switch, 0, die], [revive, revive, 0]], np.float64)
    return _viterbi(np.ascontiguousarray(logp, np.float64), trans)


def decode(proba, period, params):
    """Labels 0/1/2 inside each period, X (3) outside."""
    out = np.full(len(proba), 3, np.int8)
    logp = np.log(np.clip(proba, 1e-6, 1)) * params.get("temp", 1.0)
    for p in np.unique(period[period > 0]):
        sel = np.flatnonzero(period == p)
        out[sel] = viterbi(logp[sel], params["switch"], params["die"], params["revive"])
    return out


def score(gt, pred):
    r = pm.evaluate(gt.astype(np.int8), pred.astype(np.int8))
    return r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"]


GRID = [dict(switch=s, die=dd, revive=rv)
        for s in (4, 8, 12, 16, 24, 32, 48)
        for dd in (4, 8, 16, 24)
        for rv in (4, 8, 16, 24)]


def cv(folds=((0, 1), (2, 3), (4, 5, 6))):
    mask = bar_ath_ball_mask()
    probas = {}
    for fold in folds:
        val = [idsse.MATCHES[i] for i in fold]
        tr = [m for m in idsse.MATCHES if m not in val]
        X, y, names = dataset(tr, mask)
        model = train(X, y)
        rng = np.random.default_rng(SEED + 1)
        for m in val:
            d = add_noise(idsse.load(m), mask, rng)
            p, yv, period = predict_proba(model, d), d["y"], d["period"]
            probas[m] = (p, yv, period)
            acc = (p.argmax(1) == yv)[yv >= 0].mean()
            print("{} frame acc {:.3f}".format(m, acc), flush=True)
    results = []
    for params in GRID:
        s = []
        for m, (p, yv, period) in probas.items():
            gt = np.where(yv < 0, 3, yv)
            s.append(score(gt, decode(p, period, params)))
        s = np.array(s, float)
        results.append((s[:, 0].mean(), s[:, 1].mean(), s[:, 2].sum() / s[:, 3].sum(), params))
    results.sort(key=lambda r: -r[0])
    for r in results[:8]:
        print("spellF1 {:.4f} macroF1 {:.4f} poss_ratio {:.2f} {}".format(*r))
    raw = [score(np.where(yv < 0, 3, yv), np.where(period > 0, p.argmax(1), 3))
           for p, yv, period in probas.values()]
    print("argmax, no decoding: spellF1 {:.4f}".format(np.mean([r[0] for r in raw])))
    best = results[0][3]
    with open(DECODER, "w") as fh:
        json.dump(best, fh)
    print("wrote", DECODER, best)


def predict():
    mask = bar_ath_ball_mask()
    X, y, names = dataset(list(idsse.MATCHES), mask)
    model = train(X, y)
    d = bar_ath()
    proba = predict_proba(model, d)
    params = json.load(open(DECODER))
    labels = decode(proba, d["period"], params)
    os.makedirs(OUT, exist_ok=True)
    np.save(os.path.join(OUT, "proba.npy"), proba.astype(np.float32))
    path = os.path.join(OUT, "submission.csv")
    letters = np.array(list("HADX"))[labels]
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(letters))
    print("wrote", path, {k: int((letters == k).sum()) for k in "HADX"})


if __name__ == "__main__":
    {"cv": cv, "predict": predict}[sys.argv[1]]()
