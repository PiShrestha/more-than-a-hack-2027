"""Remove only checksum-proven duplicates from an agent-created intermediate backup."""
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def main():
    base=(Path.home()/'Documents/github/possession-research-20261116').resolve()
    expected=base/'research/artifacts'
    assert ROOT==base
    assert not expected.is_symlink()
    assert expected.resolve()==expected
    inventory=json.loads((ROOT/'research/backup-inventory.json').read_text())
    assert inventory['status']=='passed' and inventory['probability_files']==105
    files={f['path']:f for f in inventory['files']}
    removed=[]
    retained=[]
    for path in expected.rglob('*'):
        if not path.is_file() or path.is_symlink():
            continue
        rel=str(path.relative_to(expected))
        spec=files.get(rel)
        if spec is None or path.stat().st_size!=spec['bytes']:
            retained.append(rel)
            continue
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):
                h.update(block)
        if h.hexdigest()!=spec['sha256']:
            retained.append(rel)
            continue
        path.unlink()
        removed.append(dict(path=rel,bytes=spec['bytes']))
    for path in sorted(expected.rglob('*'),key=lambda p:len(p.parts),reverse=True):
        if path.is_dir() and not path.is_symlink() and not any(path.iterdir()):
            path.rmdir()
    if expected.exists() and not any(expected.iterdir()):
        expected.rmdir()
    report=dict(removed_files=len(removed),removed_bytes=sum(f['bytes'] for f in removed),
                retained_unmatched_files=retained,
                note='Only files matching a completed, locally verified backup inventory were removed; scratch inputs/results and the full Mac backup were untouched.')
    (Path(os.environ['RESEARCH_RESULTS'])/'duplicate-cleanup.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
