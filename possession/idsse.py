"""IDSSE open Bundesliga tracking (Bassek et al. 2025, CC-BY 4.0) as labelled training data.

DFL position data carries a per-frame BallStatus (alive/dead) and BallPossession (team) -- the same
two quantities the challenge ground truth is derived from, on other games. Files are fetched from
figshare by possession/fetch_idsse.sh into data/external/idsse/ and converted here to the array
layout of possession/data.py, plus `y` (0 H, 1 A, 2 D, -1 outside play), cached as .npz.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import DATA_DIR  # noqa: E402

SRC = os.path.join(DATA_DIR, "external", "idsse")
MATCHES = ("J03WMX", "J03WN1", "J03WPY", "J03WOH", "J03WQQ", "J03WOY", "J03WR9")


def _convert(match):
    from kloppy import sportec
    from kloppy.domain import BallState

    ds = sportec.load_tracking(
        meta_data=os.path.join(SRC, match + "_meta.xml"),
        raw_data=os.path.join(SRC, match + "_tracking.xml"),
        only_alive=False, coordinates="sportec")
    L, W = ds.metadata.pitch_dimensions.pitch_length, ds.metadata.pitch_dimensions.pitch_width
    home, away = ds.metadata.teams
    players = list(home.players) + list(away.players)
    col = {p.player_id: i for i, p in enumerate(players)}
    team = np.array([0] * len(home.players) + [1] * len(away.players), np.int8)

    T, S = len(ds.records), len(players)
    px = np.full((T, S), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    bx = np.full(T, np.nan, np.float32)
    by = np.full_like(bx, np.nan)
    period = np.zeros(T, np.int8)
    y = np.full(T, -1, np.int8)
    for t, fr in enumerate(ds.records):
        period[t] = fr.period.id
        if fr.ball_coordinates is not None:
            bx[t], by[t] = fr.ball_coordinates.x, fr.ball_coordinates.y
        for p, pd in fr.players_data.items():
            if pd.coordinates is not None:
                i = col[p.player_id]
                px[t, i], py[t, i] = pd.coordinates.x, pd.coordinates.y
        if fr.ball_state == BallState.DEAD:
            y[t] = 2
        elif fr.ball_owning_team is not None:
            y[t] = 0 if fr.ball_owning_team == home else 1
    # sportec coordinates are centred; move the origin to a corner like the ATD
    px += L / 2
    bx += L / 2
    py += W / 2
    by += W / 2
    return dict(px=px, py=py, team=team, bx=bx, by=by, period=period, y=y,
                pitch=np.array([L, W], np.float32))


def load(match):
    path = os.path.join(SRC, match + ".npz")
    if not os.path.exists(path):
        np.savez(path, **_convert(match))
    d = dict(np.load(path))
    d["pitch"] = tuple(float(v) for v in d["pitch"])
    return d


if __name__ == "__main__":
    for m in sys.argv[1:] or MATCHES:
        d = load(m)
        y = d["y"]
        print(m, len(y), "pitch", d["pitch"], "periods", np.unique(d["period"]),
              "H/A/D/none", [(y == k).sum() for k in (0, 1, 2, -1)],
              "ball nan %.3f" % np.isnan(d["bx"]).mean())
