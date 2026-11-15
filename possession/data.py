"""Load the tactical ATD into dense numpy arrays, cached as .npz under data/cache/.

Arrays (T = 199319 video frames, S = 46 track slots), coordinates in metres, origin at a corner:
    px, py   (T, S) float32   player positions, NaN where the slot is empty
    ps       (T, S) float32   player speed m/s
    team     (S,)   int8      0 home, 1 away
    bx, by   (T,)   float32   ball position, NaN where the ball is not tracked
    bs       (T,)   float32   ball speed as given (mostly NaN)
Row i is video frame i (ATD frame_id - 1).
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402

CACHE = os.path.join(DATA_DIR, "cache", "{}_tactical_atd.npz".format(GAME))


def _parse(raw_path, n_frames, n_slots, pitch):
    length, width = pitch
    px = np.full((n_frames, n_slots), np.nan, np.float32)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    ball = np.full((n_frames, 3), np.nan, np.float32)
    with open(raw_path) as fh:
        for line in fh:
            fid, players, b = line.rstrip("\n").split(":")
            i = int(fid) - 1
            ball[i] = [float(v) for v in b.split(",")]
            vals = np.array(players.replace(";", ",").split(","), dtype=np.float32).reshape(-1, 3)
            px[i], py[i], ps[i] = vals[:, 0], vals[:, 1], vals[:, 2]
    px *= length
    py *= width
    ball[:, 0] *= length
    ball[:, 1] *= width
    return px, py, ps, ball


def load(refresh=False):
    if os.path.exists(CACHE) and not refresh:
        return dict(np.load(CACHE))
    meta, raw = require(["atd_meta", "atd_raw"])
    team = team_of_each_track(meta)
    px, py, ps, ball = _parse(raw, source_frames("tactical"), len(team), GAMES[GAME]["pitch"])
    out = dict(px=px, py=py, ps=ps, team=team, bx=ball[:, 0], by=ball[:, 1], bs=ball[:, 2])
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez(CACHE, **out)
    return out


def bar_ath():
    """load() plus period (1/2 inside the graded halves, 0 elsewhere) and pitch, for features()."""
    from games import half_video_blocks
    d = load()
    period = np.zeros(len(d["bx"]), np.int8)
    for half, first, last in half_video_blocks(GAME):
        period[first:last + 1] = half
    d["period"] = period
    d["pitch"] = GAMES[GAME]["pitch"]
    return d


if __name__ == "__main__":
    d = load(refresh=True)
    for k, v in d.items():
        print(k, v.shape, v.dtype, "nan%.1f%%" % (100 * np.isnan(v).mean()) if v.dtype.kind == "f" else "")
