"""Repeat retained-model inference and require byte-identical research CSV output."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = Path(os.environ['RESEARCH_RESULTS']) / 'final-tree'
    before = json.loads((out / 'report.json').read_text())
    original_report = out / 'first-run-report.json'
    if not original_report.exists():
        original_report.write_text(json.dumps(before, indent=2) + '\n')
    subprocess.run([sys.executable, str(ROOT / 'research/retain_final_tree.py')], check=True)
    after = json.loads((out / 'report.json').read_text())
    for key in ('saved_tree_sha256', 'frozen_csv_sha256', 'candidate_csv_sha256'):
        if before[key] != after[key]:
            raise RuntimeError('Repeated inference changed ' + key)
    candidate = out / 'half-isolated-research-candidate.csv'
    assert hashlib.sha256(candidate.read_bytes()).hexdigest() == before['candidate_csv_sha256']
    report = dict(status='passed', repeats=2, candidate_status='research_only_not_submitted',
                  saved_tree_sha256=after['saved_tree_sha256'],
                  frozen_csv_sha256=after['frozen_csv_sha256'],
                  candidate_csv_sha256=after['candidate_csv_sha256'],
                  note='Second run reuses the saved tree; both runs execute all six stock CNNs.')
    (out / 'candidate-reproduction.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
