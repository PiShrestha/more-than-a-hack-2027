"""Fetch pinned other-match inputs atomically; verify all required data before training."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ['MTAH_DATA_DIR'])
PAIRS = '''J03WMX 51643475 51643514
J03WN1 51643472 51643517
J03WPY 51643487 51643526
J03WOH 51643478 51643520
J03WQQ 51643484 51643529
J03WOY 51643481 51643523
J03WR9 51643490 51643532'''

def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    manifest = []
    for entry in json.loads((ROOT/'research/tactical-manifest.json').read_text()):
        path = DATA/entry['path']
        assert path.stat().st_size == entry['size'], str(path)
        assert digest(path, 'sha256') == entry['sha256'], str(path)
        manifest.append(entry)
    source = json.loads((ROOT/'research/idsse-source.json').read_text())
    available = {f['id']: f for f in source['files']}
    dest = DATA/'external/idsse'
    dest.mkdir(parents=True, exist_ok=True)
    for row in PAIRS.splitlines():
        match, *ids = row.split()
        for kind, fid in zip(('meta', 'tracking'), ids):
            spec = available[int(fid)]
            path = dest/f'{match}_{kind}.xml'
            if not path.exists():
                part = path.with_suffix('.xml.part')
                subprocess.run(['curl', '--fail', '--location', '--retry', '3',
                                '--output', str(part), spec['download_url']], check=True)
                assert part.stat().st_size == spec['size'], str(part)
                assert digest(part, 'md5') == spec['computed_md5'], str(part)
                part.replace(path)
            assert path.stat().st_size == spec['size'], str(path)
            assert digest(path, 'md5') == spec['computed_md5'], str(path)
            manifest.append(dict(path=str(path.relative_to(DATA)), size=spec['size'],
                                 md5=spec['computed_md5'], sha256=digest(path, 'sha256'),
                                 source_id=int(fid)))
            print('verified', path.name, flush=True)
    (DATA/'verified-inputs.json').write_text(json.dumps(manifest, indent=2)+'\n')

if __name__ == '__main__':
    main()
