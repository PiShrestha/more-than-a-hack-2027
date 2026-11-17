# Learning possession when the ball disappears

**Team: encara_messi**  
More than a Hack 2027, Phase 1

## Approach and contribution

The ball is missing from about 41% of the scored BAR-ATH tracking frames. Many gaps last several seconds. A nearest-player rule cannot tell us much when there is no ball to measure.

Our main contribution is to train on other matches after making their tracking fail in similar ways. We reuse BAR-ATH's ball-missing pattern as an observation template, without using its possession labels. We also remove a misleading shortcut in the external data: the ball stops moving whenever play is dead.

We combine gradient-boosted trees with six temporal convolutional networks. A Viterbi decoder turns their frame predictions into possessions. The trees capture local player and ball relationships. The networks add longer context around gaps. The decoder discourages brief, unsupported changes of team.

The individual methods are established. The distinctive part is how we prepare the training data and combine them for this match's tracking failures.

## Data and assets

We use seven open matches from the Bundesliga and 2. Bundesliga in IDSSE, about one million frames at 25 frames per second. The position files supply DFL's `BallStatus` and `BallPossession` fields. These provide the external training labels: home possession, away possession, or dead ball. We did not annotate any match ourselves. We use no separate event dataset.

Source: Bassek, Rein, Weber and Memmert, *An integrated dataset of spatiotemporal and event data in elite soccer*, Scientific Data, 2025. [Paper and collection methods](https://doi.org/10.1038/s41597-025-04505-y). The dataset is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The repository downloads its position and metadata files from Figshare.

For BAR-ATH, we use the automatic tactical tracking of players and ball, plus the published half boundaries. The ball-missing pattern is calculated from tracking availability, not possession labels.

We considered automatic Smart Tagging but left it out of the final model because the external training matches lack equivalent tags. Tactical and broadcast video, homography, and broadcast tracking are also unused.

No BAR-ATH labels are used for training, parameter tuning, or model selection. The published five minutes are used only to check the pipeline. There are no manual tracking corrections or frame-specific answer rules.

## Training and prediction

We hide the external ball using BAR-ATH's missingness template at random offsets. We also hide players in bursts, with about 7% missing observations and an average outage of two seconds. We replace frozen dead-ball stretches with slow, irregular movement toward the restart position. We avoid independent position jitter because it creates unrealistic speeds.

Each frame has 111 features. These describe ball visibility and motion, gaps, distances to players, team shape, player movement, and nearby time windows. Team summaries avoid relying on persistent player IDs, which the automatic tracking does not provide reliably.

LightGBM uses 1,600 boosting rounds, 63 leaves, and every third training frame. The temporal networks use ten residual blocks with increasing dilation. Their convolutional window spans about 82 seconds. The base network has width 96 and trains for 30 epochs on 4,096-frame chunks. The six saved networks include extra seeds, more noise realizations, longer training, and a wider network.

We swap the teams during training and average both orientations at prediction time. The final mixture gives 75% weight to the mean network prediction and 25% to LightGBM.

We choose the mixture and decoder settings using three match-level folds across the seven external matches. Every held-out match is excluded from that fold's model training. The same seven matches are used for model selection, so these are validation results, not an untouched test.

## Possession rules

The decoder chooses a path through home possession (`H`), away possession (`A`), and dead ball (`D`). Its transition penalties are 4 for switching teams, 8 for entering dead ball, and 32 for returning to play.

A short opposition touch may be absorbed into the surrounding possession when its evidence does not outweigh the switch costs. This is not a strict last-touch rule. Loose balls and challenges are inferred from the learned scores and surrounding play.

A restart occurs when the path leaves `D` for `H` or `A`. We do not assign ownership during dead ball. Out-of-play decisions use learned cues such as ball position, gap timing, and player movement, rather than a fixed boundary rule.

Each half is decoded separately. Frames outside the published halves receive `X`. Every frame inside them receives `H`, `A`, or `D`. Prediction uses future observations, so this is an offline method.

## Results and limits

An independent Rivanna reconstruction obtained mean external-match spellF1 of **0.6383**, versus **0.1925** for the shipped smooth50 baseline under the same noise. The original team run reported **0.637** for the ensemble. Differences between runs should not be read as an improvement.

The original public-window check reported spellF1 of **0.715** for the ensemble and **0.326** for smooth50. That window contains only 17 possessions. It cannot establish the private full-match score.

Long ball gaps still hide turnovers. Short possessions are often merged. External possession definitions also differ from the challenge, and random placement of gaps does not reproduce their relationship with crowded play.

The networks use GroupNorm over the full input sequence. This makes predictions depend on distant context as well as the local convolutional window. Our external-match checks confirmed this sensitivity. A research change reduced it, but did not establish a general accuracy gain and is not part of this submission.

With more time, we would test the normalization change on new matches and multiple training seeds, then improve ball recovery through long gaps.

## Reproduction

Use the public [possession-model branch](https://github.com/PiShrestha/more-than-a-hack-2027/tree/possession-model), frozen at commit `602b6139b8fe82d0256515102444534bf94b32d7`. Place the permitted tactical data in `data/tactical/` as described in `DATA.md`. With Python 3.14:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r requirements-solution.txt
bash run.sh
```

The script downloads and converts IDSSE, trains LightGBM, loads the six committed networks, predicts on CPU, and validates `submission/submission.csv`. Leave the optional research commands commented out. They can replace trained weights or the decoder configuration.

Rename the output to `encara_messi.csv` for upload. It contains 199,319 frame labels. Repeated Linux runs, including a fresh-clone check, reproduced this SHA-256 exactly:

```text
557ea3435993e72a83839000e98799be75d268f91a8e1fc0432686ab9a27c19c
```
