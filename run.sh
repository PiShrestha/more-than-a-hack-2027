#!/bin/bash
# End to end: provided data in ./data/tactical + IDSSE download -> submission/submission.csv
set -euo pipefail
cd "$(dirname "$0")"
bash possession/fetch_idsse.sh
python -m possession.idsse                       # convert IDSSE XML -> cached arrays (~1 min/match)
# python -W ignore -m possession.model cv        # optional (~5 min): re-derives possession/decoder.json, which is committed
python -W ignore -m possession.model predict     # train on all 7 IDSSE matches, write the submission
python evaluation/validate_submission.py submission/submission.csv
