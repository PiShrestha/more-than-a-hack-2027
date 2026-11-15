#!/bin/bash
# End to end: provided data in ./data/tactical + IDSSE download -> submission/submission.csv
set -euo pipefail
cd "$(dirname "$0")"
bash possession/fetch_idsse.sh
python -m possession.idsse                       # convert IDSSE XML -> cached arrays (~1 min/match)
# python -W ignore -m possession.model cv        # optional: LightGBM held-out probabilities
# python -W ignore -m possession.tcn cv          # optional (GPU): TCN held-out probabilities
# python -W ignore -m possession.model ens       # optional: re-derives possession/decoder.json (mix weight + penalties), committed
# python -W ignore -m possession.tcn train       # optional (GPU, ~5 min): re-trains possession/tcn.pt, which is committed
python -W ignore -m possession.model predict     # train LightGBM on all 7 IDSSE matches, mix with the TCN, write the submission
python evaluation/validate_submission.py submission/submission.csv
