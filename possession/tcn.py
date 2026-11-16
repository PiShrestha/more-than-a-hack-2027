"""Temporal CNN over the per-frame features: ~80 s receptive field, ensembled with LightGBM.

    python -m possession.tcn cv        # same 3 folds as model.cv; writes data/external/idsse/cvtcn_<m>.npz
    python -m possession.tcn train     # train on all 7 IDSSE matches -> possession/tcn.pt (committed)
    python -m possession.tcn predict   # BAR-ATH probabilities from possession/tcn.pt, on CPU

The LightGBM windows stop at 15 s; ball gaps run 10-40 s. A dilated, non-causal CNN sees the
whole gap and both of its edges at once. Weights are committed and inference runs on CPU, so the
submission does not depend on GPU floating point.
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import idsse  # noqa: E402
from possession.data import bar_ath  # noqa: E402
from possession.features import features  # noqa: E402
from possession.model import add_noise, bar_ath_ball_mask, SEED  # noqa: E402

TAG = os.environ.get("TCN_TAG", "")  # experiment / ensemble-member suffix
WEIGHTS = os.path.join(ROOT, "possession", "tcn{}.pt".format(TAG))
TSEED = int(os.environ.get("TCN_SEED", SEED))
N_REAL = int(os.environ.get("TCN_REAL", 4))  # noise realisations per training match
CHUNK = 4096
EPOCHS = int(os.environ.get("TCN_EPOCHS", 30))
WIDTH = int(os.environ.get("TCN_WIDTH", 96))
DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)


class TCN(nn.Module):
    def __init__(self, n_in, width=WIDTH):
        super().__init__()
        self.inp = nn.Conv1d(n_in, width, 1)
        self.blocks = nn.ModuleList(
            nn.Sequential(nn.Conv1d(width, width, 3, padding=d, dilation=d), nn.GroupNorm(8, width),
                          nn.GELU(), nn.Dropout(0.1), nn.Conv1d(width, width, 1))
            for d in DILATIONS)
        self.out = nn.Conv1d(width, 3, 1)

    def forward(self, x):  # x (B, F, T)
        h = self.inp(x)
        for b in self.blocks:
            h = h + b(h)
        return self.out(h)


def _prep(X, stats):
    mu, sd = stats
    Z = (X - mu) / sd
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0).clip(-8, 8)
    return Z.astype(np.float32)


def _realisations(matches, mask, n, seed):
    """[(X, y)] per match per realisation, both orientations (teams swapped)."""
    rng = np.random.default_rng(seed)
    out = []
    for m in matches:
        for _ in range(n):
            d = add_noise(idsse.load(m), mask, rng)
            y = d["y"].astype(np.int64)
            X, _ = features(d)
            Xf, _ = features(d, flip=True)
            out.append((X, y))
            out.append((Xf, np.where(y == 0, 1, np.where(y == 1, 0, y))))
    return out


def fit(series, device, log=print):
    torch.manual_seed(TSEED)
    allX = np.concatenate([X for X, _ in series])
    stats = (np.nanmean(allX, 0), np.nanstd(allX, 0) + 1e-3)
    data = [(torch.from_numpy(_prep(X, stats)), torch.from_numpy(y)) for X, y in series]
    net = TCN(allX.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-2)
    steps_per_epoch = sum(len(y) for _, y in data) // (CHUNK * 8)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=EPOCHS * steps_per_epoch)
    gen = np.random.default_rng(TSEED)
    lossf = nn.CrossEntropyLoss(ignore_index=-1)
    for ep in range(EPOCHS):
        net.train()
        tot = 0.0
        for _ in range(steps_per_epoch):
            xb, yb = [], []
            for _ in range(8):
                X, y = data[gen.integers(len(data))]
                s = gen.integers(0, len(y) - CHUNK)
                xb.append(X[s:s + CHUNK].T)
                yb.append(y[s:s + CHUNK])
            xb, yb = torch.stack(xb).to(device), torch.stack(yb).to(device)
            loss = lossf(net(xb), yb)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tot += loss.item()
        log("epoch {} loss {:.4f}".format(ep, tot / steps_per_epoch))
    return net.cpu().eval(), stats


@torch.no_grad()
def proba(net, stats, X):
    """Full-sequence inference on CPU, symmetrised by the caller."""
    torch.set_num_threads(8)
    Z = torch.from_numpy(_prep(X, stats)).T[None]
    return torch.softmax(net(Z)[0], 0).T.numpy().astype(np.float64)


def proba_sym(net, stats, d):
    p = proba(net, stats, features(d)[0])
    pf = proba(net, stats, features(d, flip=True)[0])[:, [1, 0, 2]]
    return 0.5 * (p + pf)


def cv(folds=((0, 1), (2, 3), (4, 5, 6))):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mask = bar_ath_ball_mask()
    for fold in folds:
        val = [idsse.MATCHES[i] for i in fold]
        tr = [m for m in idsse.MATCHES if m not in val]
        net, stats = fit(_realisations(tr, mask, N_REAL, TSEED), device)
        rng = np.random.default_rng(SEED + 1)  # identical validation noise to model.cv
        for m in idsse.MATCHES:
            d = add_noise(idsse.load(m), mask, rng) if m in val else None
            if d is None:
                continue
            p = proba_sym(net, stats, d)
            y = d["y"]
            print("{} frame acc {:.3f}".format(m, (p.argmax(1) == y)[y >= 0].mean()), flush=True)
            np.savez(os.path.join(idsse.SRC, "cvtcn{}_{}.npz".format(TAG, m)), p=p, y=y,
                     period=d["period"], bvis=~np.isnan(d["bx"]))


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    net, stats = fit(_realisations(list(idsse.MATCHES), bar_ath_ball_mask(), N_REAL, TSEED), device)
    torch.save({"state": net.state_dict(), "mu": stats[0], "sd": stats[1]}, WEIGHTS)
    print("wrote", WEIGHTS)


def load_net(n_in, tag=TAG):
    ck = torch.load(os.path.join(ROOT, "possession", "tcn{}.pt".format(tag)), weights_only=False)
    net = TCN(n_in, ck["state"]["inp.weight"].shape[0])
    net.load_state_dict(ck["state"])
    return net.eval(), (ck["mu"], ck["sd"])


def predict(tags=("",)):
    """BAR-ATH probabilities, averaged over the committed ensemble members."""
    d = bar_ath()
    X, _ = features(d)
    out = []
    for tag in tags:
        net, stats = load_net(X.shape[1], tag)
        out.append(proba_sym(net, stats, d))
    return np.mean(out, 0)


if __name__ == "__main__":
    {"cv": cv, "train": train, "predict": lambda: print(predict().mean(0))}[sys.argv[1]]()
