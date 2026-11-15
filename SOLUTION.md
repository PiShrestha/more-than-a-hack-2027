# Ball possession from automatic tracking, learned on other games

Technical document for More than a Hack 2027, Phase 1.

## Summary

We train per-frame classifiers for the three states `H`, `A` and `D` (gradient-boosted trees
and a temporal CNN, averaged) on **seven open Bundesliga matches** (IDSSE, Bassek et al. 2025), whose DFL position data carries a per-frame ball status and
ball possession. Before training, the clean Bundesliga tracking is degraded to look like the
automatic BAR-ATH tracking, including BAR-ATH's own ball-missing pattern. The per-frame
probabilities are then decoded into whole possessions with a three-state Viterbi whose switching
penalties are chosen by leave-matches-out spellF1 on the Bundesliga data.

No label of any kind from BAR-ATH is used for training, tuning or model selection.

| | spellF1 | macroF1 |
|---|---|---|
| official baseline `smooth50`, IDSSE held-out, same noise | 0.182 | 0.439 |
| LightGBM alone, IDSSE held-out, argmax without decoding | 0.334 | |
| LightGBM alone, IDSSE held-out, Viterbi decoded | 0.573 | 0.886 |
| temporal CNN alone, IDSSE held-out, Viterbi decoded | 0.611 | 0.906 |
| **ensemble, IDSSE held-out, Viterbi decoded (submitted)** | **0.622** | **0.908** |
| ensemble, BAR-ATH public 5 minutes (sanity check only) | 0.715 | 0.890 |
| official baseline `smooth50`, BAR-ATH public 5 minutes | 0.326 | 0.621 |

The BAR-ATH public window has 17 possessions and is reported only to show the pipeline is
aligned. It was never used to choose between versions (see "How versions were chosen").

## Why this approach

The decisive fact about the provided tracking is not that the ball is missing on 41% of graded
frames but **how** it is missing. Short dropouts are rare. 85% of missing frames sit in gaps longer
than 10 s, 13 gaps exceed 40 s, and ball visibility decays from about 88% to 30 to 50% across each
half. Nearest-player rules have nothing to say during those stretches, so possession and ball state
have to be inferred from the 22 players alone for a large share of the match.

That is a learning problem, and learning it needs labels. Hand labels on BAR-ATH are forbidden, but
the challenge ground truth is defined as the dead-ball flag and owning team of corrected tracking,
which is exactly what DFL position data provides per frame for other games. So we learn on those
games and transfer.

The ranking metric scores whole possessions by temporal IoU, so per-frame accuracy is not enough. A
classifier that is 90% right per frame still flickers and splits possessions, which is why decoding
alone moves LightGBM's held-out spellF1 from 0.334 to 0.573.

## Data used

**Provided assets.**

- Tactical ATD (players and ball). This is the only input at prediction time.
- Ball-missing mask of the tactical ATD over graded frames. It is used as a noise template, an
  observation pattern rather than a label, so that training data has BAR-ATH's gap statistics.
- Half boundaries from `games.py`, to keep decoding inside each half.

**Tried or considered, and not used.**

- Smart Tagging. Its set-piece windows bracket restarts well in the public window, but there is no
  equivalent on the training games to learn how much to trust it, and an untested hand-set rule was
  not worth the risk.
- Homography, both videos and the broadcast ATD. Not needed for a tracking-only model, and the
  broadcast feed is not synchronised.

**External training data.** IDSSE, "An integrated dataset of synchronized spatiotemporal and event
data in elite soccer" (Bassek, Weber, Rein and Memmert, Scientific Data 2025, CC-BY 4.0). It has 7
full matches from the 2022/23 Bundesliga and 2. Bundesliga, about 1.0 M frames at 25 fps. We use
only the position data, with its per-frame `BallStatus` (alive or dead) and `BallPossession`
(team). These fields ship with the DFL data. We did not produce or alter them, and the collection
process is described in the paper. No hand labels were produced by us for any game. The files are
fetched from figshare by `possession/fetch_idsse.sh`.

## Method

### 1. Making Bundesliga tracking look like the automatic ATD

`possession/model.py:add_noise`

- **Ball dropouts.** BAR-ATH's graded ball-missing mask is laid over each training match at a
  random offset, which reproduces gap lengths and their drift through a half exactly.
- **Player dropouts.** Each player drops out as a two-state Markov chain, about 7% missing with a
  mean outage of 2 s. This gives 20.5 tracked players per frame, as in BAR-ATH.
- **Dead balls.** DFL freezes the ball position while play is dead. That is a shortcut the ATD
  never offers, since there the ball keeps being tracked while it is fetched and placed. A first
  model learned it (0.999 alive or dead accuracy whenever the ball was visible). We replace every
  dead stretch with a slow, slightly wobbling carry from where the ball died to the restart spot.
- **No position jitter.** An early version added per-frame position noise. Differentiated, it
  inflated player speeds from a median of about 1.6 to 3.8 m/s, so BAR-ATH players looked stopped
  and the model called 64% of the match dead. The ATD positions are smooth, so the jitter was
  removed. Training and BAR-ATH feature distributions now agree (median player speed 1.65 and
  1.47 m/s).

### 2. Features

`possession/features.py` builds 111 features per frame. They are track-ID-free, because ATD slots
are tracklets and not people. All are written from the home team's point of view with x oriented
along home's attacking direction, which is inferred from the data per half.

- **Ball.** Visibility, time since and until the ball was last and next seen, gap length, linearly
  interpolated position, distance to the touchlines and goal lines (negative when outside), speed,
  forward velocity and acceleration.
- **Proximity, per team.** Nearest and second-nearest distance to the ball, players within 3 m and
  10 m, and the difference of nearest distances between the teams.
- **Shape and movement, per team.** Centroid, spread, deepest and highest player, ball position
  relative to the centroid, mean and maximum player speed, and centroid velocity.
- **Context.** Centred rolling means over 0.5, 2 and 8 s, past-only and future-only means over 1
  and 4 s, and 5 and 15 s windows of overall player speed. These carry the rhythm of stoppages,
  when everyone slows down.

### 3. Classifier

LightGBM multiclass (`H`, `A`, `D`) with 1600 rounds, 63 leaves and every third frame. The data is
mirrored by swapping the teams, and predictions are symmetrised by averaging the model on the
original and the team-swapped view, so neither side gets a learned bias.

### 4. Temporal CNN

`possession/tcn.py`. The LightGBM context stops at 15 s, while ball gaps run 10 to 40 s. A
non-causal dilated 1-D CNN over the same 111 features (10 residual blocks, dilations 1 to 512,
width 96, receptive field about 80 s) sees a whole gap and both of its edges at once. It is trained
with cross-entropy on 4096-frame chunks of 4 noise realisations per match, both team orientations,
for 30 epochs on one GPU, and symmetrised like LightGBM. The trained weights (`possession/tcn.pt`,
1.5 MB) are committed and inference runs on CPU, so the submission does not depend on GPU
floating point.

### 5. Ensemble and decoding into possessions

The two models' probabilities are mixed linearly, with weight 0.75 on the CNN. The decoder is a
Viterbi path over `H`, `A` and `D` using log-probabilities, with penalties for an `H`/`A` switch,
for alive to dead, and for dead to alive. The mixing weight and penalties
(`possession/decoder.json`: switch 8, die 4, revive 32) were chosen jointly by grid search
maximising mean spellF1 on held-out IDSSE matches (3 folds over the 7 matches, with the official
`possession_metrics.py`). Both models' held-out probabilities come from the same folds and the
same validation noise. Every graded frame gets a label. We do not abstain, because under
spellF1 an `X` only repeats the previous label.

## Possession rules as implemented

The rules are learned from DFL's definitions and not hand-written. In practice:

- **Loose balls and challenges.** Possession stays with the last team until the evidence for the
  other team is strong enough, for long enough, to pay the switch penalty. Brief touches and
  deflections are therefore absorbed into the surrounding possession.
- **Restarts.** A dead stretch ends at the frame where the decoded state returns to alive, and the
  restarting team is whatever the classifier sees taking the ball. Dead-ball ownership is not
  modelled, as the challenge asks.
- **Ball out of play.** This comes from the ball crossing a line and from the collective slowdown
  of the players. The latter is the only evidence when the ball is not tracked.

## How versions were chosen

Every decision used the IDSSE held-out spellF1 or a label-free check on BAR-ATH (the predicted
dead-ball share against the 40 to 45% normal in football). The public window was looked at after
each version only to confirm the pipeline still aligned.

| version | change | IDSSE held-out spellF1 | BAR-ATH predicted D share |
|---|---|---|---|
| v1 | first model | 0.539 (inflated by the frozen-ball leak) | 42% |
| v2 | dead-ball leak removed | 0.555 | 64% (implausible) |
| v3 | position jitter removed | 0.559 | 43% |
| v4 | last-seen and next-seen features across ball gaps | 0.559 (no gain, reverted) | 40% |
| v5a | 400 to 800 rounds, or 63 to 127 leaves | 0.569 | |
| v5b | every frame instead of every third | 0.560 (no gain) | |
| v5 | 1600 rounds, wider decoder grid | 0.573 | 42.5% |
| v5c | 3 noise realisations per match for LightGBM | 0.574 (no gain) | |
| **v6** | **temporal CNN, ensembled 0.75 with v5 (submitted)** | **0.622** | **43.0%** |

## Limitations

- **Possession granularity.** DFL possessions are shorter (mean 6.7 s) than this challenge's
  (14.2 s in the public window), so a decoder tuned on DFL may switch differently from Metrica's
  ground truth. On held-out IDSSE we predict 0.73 times the true possession count. On BAR-ATH we
  predict 315 possessions with a mean of 11.5 s.
- **Long ball gaps.** Team accuracy on held-out IDSSE is 0.91 with the ball visible and 0.83
  without it (0.90 and 0.77 for LightGBM alone). Most of the remaining error is here.
- **One noise model.** The ball mask comes from BAR-ATH, but its correlation with game state, such
  as crowded boxes and aerial balls, is not reproduced. Gaps are placed at random relative to play.
- **Smart Tagging is unused.**

With more time we would try the following.

- A model over raw player trajectories (all 22 players, not team aggregates) to estimate ball
  position and owner through gaps.
- Running our own ball detector on the tactical video to shrink the gaps.
- A calibrated restart prior from the set-piece tags.

## Reproduction

Python 3.14. Prediction is CPU only, 8 to 18 minutes with 8 cores including the 2.5 GB IDSSE
download. Pinned dependencies are in
`requirements-solution.txt`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-solution.txt
# provided match data unpacked in data/tactical/ (see DATA.md)
bash run.sh          # downloads IDSSE, converts, trains, predicts, validates
```

`run.sh` writes `submission/submission.csv`. LightGBM training is deterministic (fixed seeds,
`deterministic=True` with 8 threads) and the CNN is loaded from the committed weights, so repeated
runs give a byte-identical file, also from a fresh clone, environment and download. The optional steps in `run.sh` re-derive the committed
`possession/tcn.pt` (on a GPU, which may differ in the last bits) and `possession/decoder.json`.

| file | role |
|---|---|
| `possession/data.py` | ATD to dense arrays (metres), cached |
| `possession/idsse.py` | IDSSE XML to the same arrays plus labels, via kloppy |
| `possession/features.py` | per-frame features |
| `possession/model.py` | noise model, LightGBM, Viterbi, cross-validation, ensemble, prediction |
| `possession/tcn.py`, `possession/tcn.pt` | temporal CNN and its trained weights |
| `possession/fetch_idsse.sh`, `run.sh` | download and end-to-end run |
