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
TAG = os.environ.get("POSS_TAG", "")  # experiment suffix; empty for the submission
DECODER = os.path.join(ROOT, "possession", "decoder{}.json".format(TAG))
SEED = 0
LGB = dict(objective="multiclass", num_class=3, learning_rate=0.05, num_leaves=63,
           min_data_in_leaf=200, feature_fraction=0.7, bagging_fraction=0.7, bagging_freq=1,
           lambda_l2=1.0, verbose=-1, seed=SEED, num_threads=8,
           deterministic=True, force_col_wise=True)
LGB["num_leaves"] = int(os.environ.get("POSS_LEAVES", LGB["num_leaves"]))
N_ROUNDS = int(os.environ.get("POSS_ROUNDS", 1600))
AUG = int(os.environ.get("POSS_AUG", 1))
SUBSAMPLE = int(os.environ.get("POSS_SUB", 3))  # train on every 3rd frame: neighbouring frames are near-duplicates


# -- noise model --------------------------------------------------------------------------------

def bar_ath_ball_mask():
    d = bar_ath()
    return np.isnan(d["bx"][d["period"] > 0])


def add_noise(d, ball_mask, rng):
    """Make clean IDSSE tracking look like the automatic ATD.

    ball: BAR-ATH's own graded ball-missing mask, cut at a random offset, so gap lengths and their
    drift through a half are reproduced exactly. players: each one drops out as a two-state Markov
    chain (~7% missing, mean outage ~2 s, matching 20.5 of 22 tracked). No position jitter: per-frame
    jitter differentiates into ~2.6 m/s of fake player speed, and the ATD positions are smooth.
    """
    d = dict(d)
    T, S = d["px"].shape
    # DFL freezes the ball while it is dead (zero motion on every dead frame) -- a shortcut the ATD
    # does not offer: there the ball keeps being tracked while it is fetched and placed. Replace each
    # dead stretch with a slow carry from where it died to where play restarts, plus a smooth wobble.
    bx, by = d["bx"].copy(), d["by"].copy()
    dead = d["y"] == 2
    e = np.diff(np.concatenate(([0], dead.astype(np.int8), [0])))
    for s, t in zip(np.flatnonzero(e == 1), np.flatnonzero(e == -1)):
        n = t - s
        a = np.linspace(0, 1, n)
        a = a * a * (3 - 2 * a)
        x1, y1 = (bx[t], by[t]) if t < T else (bx[s], by[s])
        wob = np.cumsum(rng.normal(0, 0.03, (2, n)), axis=1)
        bx[s:t] = bx[s] + (x1 - bx[s]) * a + wob[0]
        by[s:t] = by[s] + (y1 - by[s]) * a + wob[1]
    d["bx"], d["by"] = bx, by
    reps = int(np.ceil(T / len(ball_mask))) + 1
    m = np.tile(ball_mask, reps)
    off = rng.integers(len(ball_mask))
    bm = m[off:off + T]
    d["bx"] = np.where(bm, np.nan, bx).astype(np.float32)
    d["by"] = np.where(bm, np.nan, by).astype(np.float32)
    p_off, p_on = 0.07 / 50 / 0.93, 1 / 50
    state = np.zeros(S, bool)
    drop = np.zeros((T, S), bool)
    u = rng.random((T, S), dtype=np.float32)
    for t in range(T):
        state = np.where(state, u[t] > p_on, u[t] < p_off)
        drop[t] = state
    d["px"] = np.where(drop, np.nan, d["px"])
    d["py"] = np.where(drop, np.nan, d["py"])
    return d


def dataset(matches, ball_mask, seed=SEED, mirror=True, sub=SUBSAMPLE):
    """Stacked features/labels over matches, noisy, optionally mirrored (teams swapped)."""
    Xs, ys = [], []
    rng = np.random.default_rng(seed)
    for m in matches:
        for k in range(AUG):  # independent noise realisations; each takes a different frame phase
            d = add_noise(idsse.load(m), ball_mask, rng)
            X, names = features(d)
            y = d["y"].astype(np.int64)
            idx = np.flatnonzero(y >= 0)[k % sub::sub]
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
        for s in (1, 2, 3, 4, 6, 8, 12)
        for dd in (4, 8, 16, 32, 64, 128)
        for rv in (4, 8, 16, 32, 64, 128)]


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
            np.savez(os.path.join(idsse.SRC, "cv{}_{}.npz".format(TAG, m)), p=p, y=yv, period=period,
                     bvis=~np.isnan(d["bx"]))
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


def ens(lgb_tag="", weights=(0.0, 0.5, 0.6, 0.75, 1.0)):
    """Tune the LightGBM/TCN mixing weight jointly with the decoder on the saved CV probabilities."""
    load = lambda pre, m: np.load(os.path.join(idsse.SRC, "{}_{}.npz".format(pre, m)))
    results = []
    for w in weights:
        probas = []
        for m in idsse.MATCHES:
            a, b = load("cv" + lgb_tag, m), load("cvtcn", m)
            assert (a["y"] == b["y"]).all()
            probas.append(((1 - w) * a["p"] + w * b["p"], a["y"], a["period"]))
        for params in GRID:
            s = np.array([score(np.where(y < 0, 3, y), decode(p, per, params)) for p, y, per in probas], float)
            results.append((s[:, 0].mean(), s[:, 1].mean(), s[:, 2].sum() / s[:, 3].sum(), dict(params, tcn_weight=w)))
        best_w = max((r for r in results if r[3]["tcn_weight"] == w), key=lambda r: r[0])
        print("tcn_weight {:.2f}: best spellF1 {:.4f} macroF1 {:.4f} {}".format(w, *best_w[:2], best_w[3]), flush=True)
    best = max(results, key=lambda r: r[0])
    with open(DECODER, "w") as fh:
        json.dump(best[3], fh)
    print("wrote", DECODER, best[3])


def predict():
    mask = bar_ath_ball_mask()
    X, y, names = dataset(list(idsse.MATCHES), mask)
    model = train(X, y)
    d = bar_ath()
    proba = predict_proba(model, d)
    params = json.load(open(DECODER))
    w = params.get("tcn_weight", 0.0)
    if w > 0:
        from possession import tcn
        proba = (1 - w) * proba + w * tcn.predict()
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
    {"cv": cv, "predict": predict, "ens": ens}[sys.argv[1]]()
