from pathlib import Path
import tempfile
import unittest

from research.robustness import (agreement, corrupted_copy, duration_summary, mask_frames,
                                 profile, rows, run_lengths, sequence_summary)
from solution.predict import Config, decode, tracking_costs


class RobustnessTests(unittest.TestCase):
    def test_masks_reproducible_and_stay_in_halves(self):
        blocks = [(1, 5, 999), (2, 1200, 2000)]
        for mode in ("iid", "burst"):
            a = mask_frames(2200, blocks, .3, mode, 11, 50)
            self.assertEqual(a, mask_frames(2200, blocks, .3, mode, 11, 50))
            self.assertNotEqual(a, mask_frames(2200, blocks, .3, mode, 22, 50))
            self.assertEqual(sum(a[:5]) + sum(a[1000:1200]) + sum(a[2001:]), 0)
            self.assertEqual(sum(mask_frames(2200, blocks, 0, mode, 11, 50)), 0)

    def test_stationary_rate_and_burst_length(self):
        # Large seeded simulation detects broken Markov transition probabilities.
        n = 200000
        mask = mask_frames(n, [(1, 0, n - 1)], .3, "burst", 33, 50)
        self.assertAlmostEqual(sum(mask) / n, .3, delta=.025)
        lengths = run_lengths(mask)
        self.assertAlmostEqual(sum(lengths) / len(lengths), 50, delta=6)

    def test_invalid_mask_parameters(self):
        for rate in (-.1, 1, float("nan")):
            with self.assertRaises(ValueError):
                mask_frames(5, [(1, 0, 4)], rate, "burst", 1, 50)
        with self.assertRaises(ValueError):
            mask_frames(5, [(1, 0, 4)], .9, "burst", 1, 1)

    def test_corruption_preserves_natural_gaps_players_and_source(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            raw, target = root / "in.txt", root / "out.txt"
            original = ("1:0.1,0.2,0;0.9,0.2,0:0.1,0.2,0\n"
                        "2:0.1,0.2,0;0.9,0.2,0:NaN,NaN,NaN\n"
                        "3:0.1,0.2,0;0.9,0.2,0:0.9,0.2,0\n")
            raw.write_text(original)
            self.assertEqual(corrupted_copy(raw, target, bytearray([1, 1, 0]), 3, 2), 1)
            self.assertEqual(raw.read_text(), original)
            output = list(rows(target, 3, 2))
            self.assertEqual([entry[2] for entry in output], [None, None, (.9, .2)])
            self.assertTrue(all(entry[1][1] == "0.1,0.2,0;0.9,0.2,0" for entry in output))
            costs, _ = tracking_costs(target, [0, 1], 3, (100, 50), 25, Config())
            self.assertEqual([column[0] for column in costs], [0, 0, 0])
            self.assertEqual(decode(costs, [(1, 0, 2)], Config()), ["A"] * 3)

    def test_missing_runs_reset_between_halves(self):
        with tempfile.TemporaryDirectory() as temporary:
            raw = Path(temporary) / "in.txt"
            raw.write_text("".join("{}:0,0,0;1,1,0:NaN,NaN,NaN\n".format(i)
                                   for i in range(1, 7)))
            report, _ = profile(raw, [0, 1], 6, [(1, 0, 1), (2, 4, 5)], 2)
            self.assertEqual([h["missing_runs"]["max_seconds"] for h in report["halves"]], [1, 1])

    def test_stability_denominators_and_no_cross_half_transition(self):
        blocks = [(1, 1, 2), (2, 4, 5)]
        clean = ["X", "H", "H", "X", "A", "A", "X"]
        other = ["X", "H", "D", "X", "A", "A", "X"]
        result = agreement(clean, other, blocks, bytearray([0, 0, 1, 0, 0, 0, 0]))
        self.assertEqual(result["all"]["changed_fraction"], .25)
        self.assertEqual(result["masked"]["changed_fraction"], 1)
        summary = sequence_summary(clean, blocks, 1)
        self.assertEqual(summary["state_transitions"], 0)
        self.assertEqual(summary["possession_durations"]["runs"], 2)
        self.assertEqual(summary["label_counts_in_halves"], {"H": 2, "A": 2})
        empty = agreement(clean, clean, blocks, bytearray(7))
        self.assertIsNone(empty["masked"]["changed_fraction"])

    def test_duration_quantiles_and_empty_input(self):
        self.assertEqual(run_lengths([True, True, False, True]), [2, 1])
        self.assertEqual(duration_summary([1, 2, 10], 2)["p90_seconds"], 5)
        self.assertEqual(duration_summary([], 25)["max_seconds"], 0)


if __name__ == "__main__":
    unittest.main()
