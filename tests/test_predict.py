import csv
import itertools
import math
from pathlib import Path
import tempfile
import unittest

from solution.predict import Config, apply_tags, decode, metadata, tracking_costs, write_csv


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = Config()

    def raw(self, text, n):
        path = self.root / "tracking.txt"
        path.write_text(text)
        return tracking_costs(path, [0, 1], n, (100, 50), 25, self.config)

    def test_missing_ball_is_neutral_not_dead(self):
        costs, info = self.raw("1:0.1,0.2,0;0.9,0.2,0:NaN,NaN,NaN\n", 1)
        self.assertEqual([list(c) for c in costs], [[0], [0], [0]])
        self.assertEqual(info["ball_observed_frames"], 0)

    def test_visible_control_and_outside_pitch(self):
        costs, _ = self.raw("1:0.1,0.2,0;0.9,0.2,0:0.11,0.2,0\n"
                            "2:0.1,0.2,0;0.9,0.2,0:1.1,0.2,0\n", 2)
        self.assertLess(costs[0][0], costs[1][0])
        self.assertLess(costs[2][1], costs[0][1])

    def test_missing_player_and_ball_axis(self):
        costs, _ = self.raw("1:NaN,NaN,0;0.9,0.2,0:0.9,0.2,0\n"
                            "2:0.1,0.2,0;0.9,0.2,0:0.9,NaN,0\n", 2)
        self.assertEqual([c[0] for c in costs], [4, 0, 1.5])
        self.assertEqual([c[1] for c in costs], [0, 0, 0])

    def test_reject_duplicate_truncated_and_bad_slot_rows(self):
        rows = "1:0,0,0;1,1,0:0,0,0\n"
        for text, n in [(rows + rows, 2), (rows, 2),
                        ("1:0,0,0:0,0,0\n", 1)]:
            with self.subTest(text=text, n=n), self.assertRaises(ValueError):
                self.raw(text, n)

    def test_occlusion_preserves_a_possession(self):
        costs = [[0] * 60, [4] * 60, [2] * 60]
        for s in range(3):
            costs[s][20:40] = [0] * 20
        self.assertEqual(decode(costs, [(1, 0, 59)], self.config), ["H"] * 60)

    def test_one_frame_deflection_is_not_a_handover(self):
        costs = [[0] * 60, [4] * 60, [2] * 60]
        costs[0][30], costs[1][30] = 4, 0
        self.assertEqual(decode(costs, [(1, 0, 59)], self.config), ["H"] * 60)

    def test_sustained_handover_and_stoppage(self):
        costs = [[0] * 30 + [4] * 60,
                 [4] * 30 + [0] * 30 + [4] * 30,
                 [2] * 60 + [0] * 30]
        self.assertEqual(decode(costs, [(1, 0, 89)], self.config),
                         ["H"] * 30 + ["A"] * 30 + ["D"] * 30)

    def test_halves_reset_and_ungraded_frames_are_x(self):
        costs = [[0] * 5 + [3] * 5, [3] * 5 + [0] * 5, [2] * 10]
        self.assertEqual(decode(costs, [(1, 1, 3), (2, 6, 8)], self.config),
                         ["X"] + ["H"] * 3 + ["X"] * 2 + ["A"] * 3 + ["X"])

    def test_viterbi_matches_exhaustive_search(self):
        costs = [[0, 3, 1, 0, 4], [3, 0, 1, 4, 0], [2, 2, 0, 2, 2]]
        config = Config(switch_cost=1.25, stoppage_cost=0.75)
        def objective(path):
            total = sum(costs[s][i] for i, s in enumerate(path))
            for a, b in zip(path, path[1:]):
                if a != b:
                    total += config.stoppage_cost if 2 in (a, b) else config.switch_cost
            return total
        result = decode(costs, [(1, 0, 4)], config)
        codes = ["HAD".index(s) for s in result]
        self.assertAlmostEqual(objective(codes), min(map(objective, itertools.product(range(3), repeat=5))))

    def test_tags_reverse_defending_team_ignore_restarts_and_cap_overlap(self):
        def event(code, team, start, end):
            return ("<instance><code>{}</code><start>{}</start><end>{}</end>"
                    "<label><group>Team</group><text>{}</text></label></instance>"
                    .format(code, start, end, team))
        path = self.root / "tags.xml"
        path.write_text("<root>" + event("BUILD UP", "Barcelona", 0, 1) * 2
                        + event("DEFENDING IN MIDDLE THIRD", "Barcelona", 1, 2)
                        + event("SET PIECES", "Barcelona", 2, 3)
                        + event("BUILD UP", "N/A", 2, 3) + "</root>")
        costs = [[0.] * 75 for _ in range(3)]
        self.assertEqual(apply_tags(costs, path, 25, 0.35), 3)
        self.assertEqual([c[0] for c in costs], [0, 0.35, 0.35])
        self.assertEqual([c[25] for c in costs], [0.35, 0, 0.35])
        self.assertEqual([c[50] for c in costs], [0, 0, 0])

    def test_metadata_rejects_unknown_team(self):
        path = self.root / "meta.xml"
        path.write_text('<root><Team id="h"/><Team id="a"/>'
                        '<Player teamId="other"/></root>')
        with self.assertRaises(ValueError):
            metadata(path)

    def test_invalid_parameters_and_costs(self):
        for values in ({"distance_scale": 0}, {"switch_cost": -1}, {"tag_weight": math.nan}):
            with self.assertRaises(ValueError):
                Config(**values)
        with self.assertRaises(ValueError):
            decode([[0], [0], [math.nan]], [(1, 0, 0)], self.config)

    def test_csv_is_deterministic_and_contiguous(self):
        path = self.root / "predictions.csv"
        labels = ["X", "H", "A", "D"]
        write_csv(path, labels)
        first = path.read_bytes()
        write_csv(path, labels)
        self.assertEqual(path.read_bytes(), first)
        with path.open(newline="") as handle:
            self.assertEqual(list(csv.reader(handle)),
                             [["frame", "label"], ["0", "X"], ["1", "H"], ["2", "A"], ["3", "D"]])


if __name__ == "__main__":
    unittest.main()
