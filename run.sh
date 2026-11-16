#!/bin/bash
# End to end: provided data in ./data/tactical + IDSSE download -> submission/submission.csv
set -euo pipefail
cd "$(dirname "$0")"
bash possession/fetch_idsse.sh
python -m possession.idsse                       # convert IDSSE XML -> cached arrays (~1 min/match)
# python -W ignore -m possession.model cv        # optional: LightGBM held-out probabilities
# python -W ignore -m possession.tcn cv          # optional (GPU): TCN held-out probabilities
# python -W ignore -m possession.model ens       # optional: re-derives possession/decoder.json (mix weight + penalties), committed
# optional (GPU, ~5 min each): re-train the six committed CNNs possession/tcn*.pt
# python -W ignore -m possession.tcn train
# for v in r8:TCN_REAL=8 e60:TCN_EPOCHS=60 w128:TCN_WIDTH=128 s1:TCN_SEED=1 s2:TCN_SEED=2; do
#   env TCN_TAG=_${v%%:*} ${v#*:} python -W ignore -m possession.tcn train; done
python -W ignore -m possession.model predict     # train LightGBM on all 7 IDSSE matches, mix with the TCN, write the submission
python evaluation/validate_submission.py submission/submission.csv
