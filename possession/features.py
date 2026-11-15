"""Per-frame, track-ID-free features for possession / ball-state classification.

Input is a dict of dense arrays in metres (see possession/data.py):
    px, py (T, S), team (S,) 0 home / 1 away, bx, by (T,), period (T,) int, 0 = not in play
plus pitch (L, W). Features are written from the HOME team's point of view, with x oriented along
home's attacking direction, so a model trained on them is direction-invariant. features(..., flip=True)
recomputes them with the teams swapped, which is how training data is mirrored and how predictions
are symmetrised.
"""

import numpy as np

FPS = 25


# -- nan-aware rolling windows via cumulative sums --------------------------------------------

def _cum(x):
    ok = ~np.isnan(x)
    return (np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0)))),
            np.concatenate(([0], np.cumsum(ok))))


def roll_mean(x, w, mode="center"):
    """Mean of non-NaN values in a window of w frames: centred, past (t-w..t) or future (t..t+w)."""
    s, c = _cum(x.astype(np.float64))
    n = len(x)
    i = np.arange(n)
    if mode == "center":
        lo, hi = i - w // 2, i + w // 2 + 1
    elif mode == "past":
        lo, hi = i - w + 1, i + 1
    else:
        lo, hi = i, i + w
    lo, hi = np.clip(lo, 0, n), np.clip(hi, 0, n)
    cnt = c[hi] - c[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(cnt > 0, (s[hi] - s[lo]) / np.maximum(cnt, 1), np.nan).astype(np.float32)


def _gap_lengths(missing):
    """Frames since last observed and until next observed (0 when observed)."""
    n = len(missing)
    idx = np.arange(n)
    last = np.maximum.accumulate(np.where(~missing, idx, -10 ** 6))
    nxt = np.minimum.accumulate(np.where(~missing, idx, 10 ** 7)[::-1])[::-1]
    return np.minimum(idx - last, 10 ** 4), np.minimum(nxt - idx, 10 ** 4)


def _interp(v, period):
    """Linear interpolation of NaNs inside each period; edges held."""
    out = v.copy()
    for p in np.unique(period[period > 0]):
        sel = np.flatnonzero(period == p)
        seg = v[sel]
        ok = ~np.isnan(seg)
        if ok.sum() >= 2:
            out[sel] = np.interp(np.arange(len(seg)), np.flatnonzero(ok), seg[ok])
    return out


def attack_sign(d):
    """(T,) +1 where home attacks toward x = L, -1 otherwise. Data-driven: home's mean x per period."""
    L = d["pitch"][0]
    sign = np.ones(len(d["bx"]), np.float32)
    home = d["team"] == 0
    for p in np.unique(d["period"][d["period"] > 0]):
        sel = d["period"] == p
        hx = np.nanmean(d["px"][sel][:, home])
        sign[sel] = 1.0 if hx < L / 2 else -1.0
    return sign


def features(d, flip=False):
    L, W = d["pitch"]
    team = (1 - d["team"]) if flip else d["team"]
    px, py, period = d["px"], d["py"], d["period"]
    sign = attack_sign(d) * (-1 if flip else 1)

    # orient: x' increases toward the (perspective) home team's attacking goal; y mirrored with it
    def ox(x):
        return np.where(sign[:, None] > 0, x, L - x) if x.ndim == 2 else np.where(sign > 0, x, L - x)

    def oy(y):
        return np.where(sign[:, None] > 0, y, W - y) if y.ndim == 2 else np.where(sign > 0, y, W - y)

    PX, PY = ox(px), oy(py)
    bmiss = np.isnan(d["bx"])
    BX, BY = _interp(ox(d["bx"]), period), _interp(oy(d["by"]), period)
    since, until = _gap_lengths(bmiss)

    f = {}
    f["ball_vis"] = (~bmiss).astype(np.float32)
    f["ball_since"] = np.log1p(since).astype(np.float32)
    f["ball_until"] = np.log1p(until).astype(np.float32)
    f["ball_gap"] = np.log1p(np.where(bmiss, since + until, 0)).astype(np.float32)
    f["bx"], f["by"] = BX, BY
    f["b_out"] = np.minimum(np.minimum(BX, L - BX), np.minimum(BY, W - BY))  # <0 outside the pitch
    f["b_out_x"] = np.minimum(BX, L - BX)
    f["b_out_y"] = np.minimum(BY, W - BY)
    vx = np.gradient(BX) * FPS
    vy = np.gradient(BY) * FPS
    bspd = np.hypot(vx, vy)
    f["b_spd"] = np.minimum(bspd, 50)
    f["b_vx"] = np.clip(vx, -40, 40)
    f["b_acc"] = np.minimum(np.abs(np.gradient(roll_mean(bspd, 3))) * FPS, 500)

    dist = np.hypot(PX - BX[:, None], PY - BY[:, None])
    for k, t in (("h", 0), ("a", 1)):
        m = team == t
        dk = np.sort(np.where(np.isnan(dist[:, m]), np.inf, dist[:, m]), axis=1)
        f["d1" + k] = np.minimum(dk[:, 0], 60).astype(np.float32)
        f["d2" + k] = np.minimum(dk[:, 1], 60).astype(np.float32)
        f["n3" + k] = (dk < 3).sum(1).astype(np.float32)
        f["n10" + k] = (dk < 10).sum(1).astype(np.float32)
        xs, ys = PX[:, m], PY[:, m]
        with np.errstate(all="ignore"):
            f["cx" + k] = np.nanmean(xs, 1)
            f["cy" + k] = np.nanmean(ys, 1)
            f["sx" + k] = np.nanstd(xs, 1)
            f["sy" + k] = np.nanstd(ys, 1)
            f["xmax" + k] = np.nanmax(xs, 1)
            f["xmin" + k] = np.nanmin(xs, 1)
        f["cnt" + k] = (~np.isnan(xs)).sum(1).astype(np.float32)
        # player speeds from positions (more complete than the provided speed column)
        with np.errstate(all="ignore"):
            sp = np.hypot(np.gradient(xs, axis=0), np.gradient(ys, axis=0)) * FPS
            sp[sp > 12] = np.nan
            f["spd" + k] = np.nanmean(sp, 1)
            f["spdmax" + k] = np.nanmax(sp, 1)
        f["bxrel" + k] = BX - f["cx" + k]
    f["d_diff"] = f["d1a"] - f["d1h"]
    f["spd_all"] = (f["spdh"] * f["cnth"] + f["spda"] * f["cnta"]) / np.maximum(f["cnth"] + f["cnta"], 1)
    f["cx_diff"] = f["cxh"] - f["cxa"]
    f["vcxh"] = np.gradient(roll_mean(f["cxh"], 25)) * FPS
    f["vcxa"] = np.gradient(roll_mean(f["cxa"], 25)) * FPS

    # temporal context
    ctx = {}
    for k in ("d_diff", "d1h", "d1a", "ball_vis", "b_spd", "spd_all", "b_out", "n3h", "n3a", "spdh", "spda"):
        for w in (13, 51, 201):
            ctx["{}_c{}".format(k, w)] = roll_mean(f[k], w)
    for k in ("d_diff", "n3h", "n3a", "b_spd", "spd_all", "ball_vis", "b_out"):
        for w in (25, 100):
            ctx["{}_p{}".format(k, w)] = roll_mean(f[k], w, "past")
            ctx["{}_f{}".format(k, w)] = roll_mean(f[k], w, "future")
    ctx["b_out_min_c101"] = -roll_mean(-np.minimum(f["b_out"], 5), 101)  # crude; mean of clipped
    for w in (125, 375):
        ctx["spd_all_c{}".format(w)] = roll_mean(f["spd_all"], w)
        ctx["b_spd_max_proxy_c{}".format(w)] = roll_mean((f["b_spd"] > 8).astype(np.float32), w)
    f.update(ctx)

    names = sorted(f)
    X = np.stack([np.asarray(f[n], np.float32) for n in names], 1)
    X[~np.isfinite(X)] = np.nan
    return X, names
