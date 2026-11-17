"""Release checks fail on wrong frames, changed weights, unsafe paths and mixed packages."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from tools.submission import check_environment, inside, package, reproduce, sha256, validate_csv, verify_release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'submission').mkdir()
        (self.root / 'weights.pt').write_bytes(b'fixed checkpoint')
        self.csv = self.root / 'submission/submission.csv'
        self.csv.write_text('frame,label\n0,H\n1,D\n2,X\n')
        doc = self.root / 'submission/encara_messi.md'
        doc.write_text('# Method\n')
        self.record = dict(team='encara_messi', rows=3,
                           pipeline_files={'weights.pt': sha256(self.root / 'weights.pt')},
                           artifacts={'csv': dict(path='submission/submission.csv', sha256=sha256(self.csv)),
                                      'document': dict(path='submission/encara_messi.md', sha256=sha256(doc))})
        self.save()

    def save(self):
        (self.root / 'submission/manifest.json').write_text(json.dumps(self.record))

    def test_changed_checkpoint_blocks_packaging(self):
        (self.root / 'weights.pt').write_bytes(b'changed checkpoint')
        with self.assertRaisesRegex(ValueError, 'Frozen pipeline changed'):
            package(self.root, self.root / 'out')
        self.assertFalse((self.root / 'out').exists())

    def test_frame_alignment_is_checked_even_when_hash_matches(self):
        self.csv.write_text('frame,label\n0,H\n2,D\n1,X\n')
        self.record['artifacts']['csv']['sha256'] = sha256(self.csv)
        self.save()
        with self.assertRaisesRegex(ValueError, 'Invalid CSV row'):
            verify_release(self.root)

    def test_conflict_preserves_both_destination_files(self):
        out = self.root / 'out'
        out.mkdir()
        (out / 'encara_messi.md').write_text('team edits')
        with self.assertRaisesRegex(ValueError, 'Preserving'):
            package(self.root, out)
        self.assertFalse((out / 'encara_messi.csv').exists())
        self.assertEqual((out / 'encara_messi.md').read_text(), 'team edits')

    def test_reproduced_csv_must_match_before_any_files_are_written(self):
        changed = self.root / 'candidate.csv'
        changed.write_text('frame,label\n0,A\n1,D\n2,X\n')
        with self.assertRaisesRegex(ValueError, 'Reproduced CSV differs'):
            package(self.root, self.root / 'out', changed)
        self.assertFalse((self.root / 'out').exists())

    def test_path_traversal_and_extra_rows_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'escapes'):
            inside(self.root, '../outside')
        self.csv.write_text('frame,label\n0,H\n1,D\n2,X\n3,A\n')
        with self.assertRaisesRegex(ValueError, 'Expected 3 rows'):
            validate_csv(self.csv, 3)

    def test_package_is_repeatable_and_copies_exact_document(self):
        out = self.root / 'out'
        package(self.root, out)
        first = {p.name: p.read_bytes() for p in out.iterdir()}
        package(self.root, out)
        self.assertEqual(first, {p.name: p.read_bytes() for p in out.iterdir()})
        self.assertEqual(first['encara_messi.csv'], self.csv.read_bytes())

    def test_model_environment_overrides_are_rejected_before_training(self):
        with mock.patch.dict('os.environ', {'POSS_ROUNDS': '1'}):
            with self.assertRaisesRegex(ValueError, 'Unset model overrides'):
                check_environment()

    def test_reproduction_uses_temporary_sources_and_fresh_caches(self):
        data = self.root / 'private-data'
        data.mkdir()
        raw = data / 'tracking.xml'
        raw.write_text('raw input')
        entry = dict(path='tracking.xml', size=raw.stat().st_size, sha256=sha256(raw))
        (self.root / 'submission/input-manifest.json').write_text(json.dumps([entry]))
        (data / 'cache').mkdir()
        (data / 'cache/stale.npz').write_bytes(b'stale')
        seen = []

        def pipeline(command, cwd, env, check):
            stage = Path(cwd)
            seen.append(stage)
            self.assertNotEqual(stage, self.root)
            self.assertTrue((stage / 'data/tracking.xml').is_symlink())
            self.assertFalse((stage / 'data/cache').exists())
            self.assertEqual(env['MTAH_DATA_DIR'], str(stage / 'data'))
            self.assertEqual((stage / 'weights.pt').read_bytes(), b'fixed checkpoint')
            self.assertEqual(command, ['bash', 'run.sh'])
            (stage / 'submission').mkdir()
            (stage / 'submission/submission.csv').write_bytes(self.csv.read_bytes())

        with mock.patch('tools.submission.check_environment'), mock.patch('tools.submission.subprocess.run', side_effect=pipeline):
            reproduce(self.root, data, self.root / 'out')
        self.assertFalse(seen[0].exists())
        self.assertEqual(raw.read_text(), 'raw input')
        self.assertEqual((data / 'cache/stale.npz').read_bytes(), b'stale')


if __name__ == '__main__':
    unittest.main()
