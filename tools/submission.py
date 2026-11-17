"""Verify, package, or reproduce the fixed entry. Never reads BAR-ATH ground truth."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError('Manifest path escapes the repository: ' + name)
    return path


def manifest(root):
    return json.loads((root / 'submission/manifest.json').read_text())


def validate_csv(path, rows):
    counts = dict.fromkeys('HADX', 0)
    with Path(path).open(newline='', encoding='utf-8') as stream:
        reader = csv.reader(stream)
        if next(reader, None) != ['frame', 'label']:
            raise ValueError('CSV header must be frame,label')
        total = 0
        for frame, row in enumerate(reader):
            if len(row) != 2 or row[0] != str(frame) or row[1] not in counts:
                raise ValueError('Invalid CSV row at frame ' + str(frame))
            counts[row[1]] += 1
            total += 1
    if total != rows:
        raise ValueError(f'Expected {rows} rows, found {total}')
    return counts


def verify_release(root=ROOT):
    record = manifest(root)
    for name, expected in record['pipeline_files'].items():
        if sha256(inside(root, name)) != expected:
            raise ValueError('Frozen pipeline changed: ' + name)
    for entry in record['artifacts'].values():
        if sha256(inside(root, entry['path'])) != entry['sha256']:
            raise ValueError('Release file changed: ' + entry['path'])
    csv_path = inside(root, record['artifacts']['csv']['path'])
    counts = validate_csv(csv_path, record['rows'])
    return record, counts


def package(root, output, csv_source=None):
    record, _ = verify_release(root)
    sources = [(csv_source or inside(root, record['artifacts']['csv']['path']), output / (record['team'] + '.csv')),
               (inside(root, record['artifacts']['document']['path']), output / (record['team'] + '.md'))]
    # Check every destination first. Never leave a mixed old/new package after a conflict.
    for source, target in sources:
        if target.exists() and sha256(target) != sha256(source):
            raise ValueError('Preserving a different existing file: ' + str(target))
    if sha256(sources[0][0]) != record['artifacts']['csv']['sha256']:
        raise ValueError('Reproduced CSV differs from the fixed submission')
    validate_csv(sources[0][0], record['rows'])
    output.mkdir(parents=True, exist_ok=True)
    for source, target in sources:
        if not target.exists():
            shutil.copyfile(source, target)
        if sha256(source) != sha256(target):
            raise ValueError('Copy verification failed: ' + str(target))
    return sources


def check_inputs(root, data):
    records = json.loads((root / 'submission/input-manifest.json').read_text())
    for entry in records:
        path = inside(data, entry['path'])
        if not path.is_file() or path.stat().st_size != entry['size'] or sha256(path) != entry['sha256']:
            raise ValueError('Missing or different input: ' + str(path))
    return records


def check_environment():
    overrides = sorted(name for name in os.environ if name.startswith(('POSS_', 'TCN_')))
    if overrides:
        raise ValueError('Unset model overrides before reproduction: ' + ', '.join(overrides))
    if sys.version_info[:2] != (3, 14):
        raise ValueError('Reproduction requires Python 3.14; verification and packaging use only the standard library')
    for line in (ROOT / 'requirements-solution.txt').read_text().splitlines():
        name, version = line.strip().split('==')
        installed = importlib.metadata.version(name)
        # Official CUDA/CPU wheel suffixes do not change the pinned public version.
        if installed.split('+')[0] != version:
            raise ValueError(f'{name} must be {version}, found {installed}')


def reproduce(root, data, output):
    record, _ = verify_release(root)
    check_environment()
    inputs = check_inputs(root, data)
    # Use fresh caches and a temporary source copy. The release CSV is never overwritten.
    with tempfile.TemporaryDirectory(prefix='encara-messi-reproduce-') as temporary:
        stage = Path(temporary)
        for name in record['pipeline_files']:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(inside(root, name), target)
        for entry in inputs:
            target = stage / 'data' / entry['path']
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(inside(data, entry['path']))
        env = dict(os.environ, MTAH_DATA_DIR=str(stage / 'data'), PYTHONDONTWRITEBYTECODE='1')
        env['PATH'] = str(Path(sys.executable).parent) + os.pathsep + env.get('PATH', '')
        for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[key] = '8'
        subprocess.run(['bash', 'run.sh'], cwd=stage, env=env, check=True)
        return package(root, output, stage / 'submission/submission.csv')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('verify', help='Check frozen code, weights, CSV and final document without ML dependencies')
    for name in ('package', 'reproduce'):
        command = commands.add_parser(name)
        command.add_argument('--output-dir', type=Path, required=True)
        if name == 'reproduce':
            command.add_argument('--data-dir', type=Path, required=True, help='Directory containing tactical/ and external/idsse/')
    check = commands.add_parser('check-inputs')
    check.add_argument('--data-dir', type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == 'verify':
            record, counts = verify_release()
            print(f"Verified {record['team']}: {record['rows']} frames; labels {counts}")
            print('CSV SHA-256:', record['artifacts']['csv']['sha256'])
            print('Document SHA-256:', record['artifacts']['document']['sha256'])
        elif args.command == 'check-inputs':
            verify_release()
            print('Verified', len(check_inputs(ROOT, args.data_dir.resolve())), 'required raw inputs')
        else:
            output = args.output_dir.resolve()
            if args.command == 'reproduce':
                files = reproduce(ROOT, args.data_dir.resolve(), output)
            else:
                files = package(ROOT, output)
            print('Verified upload files:')
            for _, path in files:
                print(path)
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError, subprocess.CalledProcessError) as error:
        parser.exit(1, 'ERROR: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
