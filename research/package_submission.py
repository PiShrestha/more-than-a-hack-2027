"""Stage exact submission copies locally; never edit the frozen branch or submit a form."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

ROOT=Path(__file__).resolve().parents[1]
EXPECTED='557ea3435993e72a83839000e98799be75d268f91a8e1fc0432686ab9a27c19c'
SOURCE='602b6139b8fe82d0256515102444534bf94b32d7'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--team-name',help='Exact registered name; omit to create unsubmitted review staging')
    args=parser.parse_args()
    team=args.team_name
    if team is not None:
        if not team.strip() or team!=team.strip() or team in ('.','..') or any(c in team for c in '/\\') or any(ord(c)<32 for c in team):
            parser.error('Team name must be a nonempty safe filename without surrounding whitespace or path separators')
    dest=ROOT/'research/artifacts/submission-packages'/(team or 'review-staging')
    dest.mkdir(parents=True,exist_ok=True)
    csv=dest/((team+'.csv') if team else 'submission.csv')
    doc=dest/((team+'.md') if team else 'SOLUTION.md')
    source_csv=ROOT/'submission/submission.csv'
    if sha(source_csv)!=EXPECTED:
        raise RuntimeError('Frozen submission hash differs; packaging aborted')
    for source,target in ((source_csv,csv),(ROOT/'SOLUTION.md',doc)):
        if target.exists() and sha(target)!=sha(source):
            raise RuntimeError('Preserving an edited existing package file: '+str(target))
        if not target.exists():
            shutil.copyfile(source,target)
    checked=subprocess.run([sys.executable,str(ROOT/'evaluation/validate_submission.py'),str(csv)],
                           capture_output=True,text=True,check=True,cwd=ROOT)
    (dest/'validation.txt').write_text(checked.stdout)
    deadline=datetime(2026,11,17,23,59,tzinfo=ZoneInfo('Europe/Madrid'))
    instructions=ROOT/'research/artifacts/submission-packages/organizer-instructions.json'
    private_details=json.loads(instructions.read_text()) if instructions.exists() else {}
    manifest=dict(status='staged_for_team_review',team_name=team,
        source_commit=SOURCE,repo_branch_url='https://github.com/PiShrestha/more-than-a-hack-2027/tree/possession-model',
        repo_commit_url='https://github.com/PiShrestha/more-than-a-hack-2027/tree/'+SOURCE,
        submission_form=private_details.get('submission_form'),deadline_cet=deadline.isoformat(),
        deadline_new_york=deadline.astimezone(ZoneInfo('America/New_York')).isoformat(),
        csv=dict(filename=csv.name,sha256=sha(csv),rows=199319),
        technical_document=dict(filename=doc.name,sha256=sha(doc),version='unchanged frozen draft; team review still required'),
        outstanding=['Confirm registered team name' if team is None else 'Team name supplied; verify it matches every registration',
                     'Team reviews technical document and proposed review patch',
                     'Designate one submitter; last response per team before the deadline is scored'],
        form_submitted=False,
        deadline_source='Official organizer submission-instructions email dated November 13, 2026')
    (dest/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Staged:',dest)
    print('CSV hash verified:',EXPECTED)
    print('Deadline in New York:',manifest['deadline_new_york'])
    print('Form not submitted; team/document review remains')

if __name__=='__main__':
    main()
