"""Infer H/A/D possession with distance evidence, tactical priors and sequence decoding.

Run from the repository root: python -m solution.predict --out evaluation/results/sequence.csv
The public ground truth is never opened by this module.
"""

import argparse
from array import array
import csv
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from games import GAME, GAMES, asset_paths, half_video_blocks

STATES = ("H", "A", "D")
ATTACK = {"build up", "progression", "creating chances", "attacking transition",
          "ball in final third", "ball in the box", "long ball"}
DEFEND = {"defending in defensive third", "defending in middle third",
          "defending in attacking third", "defensive transition"}


@dataclass(frozen=True)
class Config:
    # Physical scales in metres, seconds, and additive negative-log-like costs.
    # These are initial heuristics, not fitted to the public five-minute labels.
    control_radius: float = 2.5
    distance_scale: float = 2.0
    switch_cost: float = 12.0
    stoppage_cost: float = 8.0
    outside_margin: float = 0.5
    stationary_seconds: float = 2.0
    stationary_radius: float = 0.75
    tag_weight: float = 0.35

    def __post_init__(self):
        for name, value in asdict(self).items():
            if not math.isfinite(value) or value < 0:
                raise ValueError("{} must be finite and nonnegative".format(name))
        if self.distance_scale == 0 or self.stationary_seconds == 0:
            raise ValueError("distance_scale and stationary_seconds must be positive")


def metadata(path):
    root = ET.parse(path).getroot()
    teams = list(root.iter("Team"))
    if len(teams) != 2 or len({t.get("id") for t in teams}) != 2:
        raise ValueError("ATD metadata must define two distinct teams")
    ids = {t.get("id"): i for i, t in enumerate(teams)}
    slots = []
    for player in root.iter("Player"):
        team_id = player.get("teamId")
        if team_id not in ids:
            raise ValueError("Unknown teamId in tracking metadata: {!r}".format(team_id))
        slots.append(ids[team_id])
    if not slots:
        raise ValueError("No tracking slots in metadata")
    return slots


def xy(group):
    fields = group.split(",")
    if len(fields) < 2:
        raise ValueError("Expected x,y[,speed] tracking group")
    point = (float(fields[0]), float(fields[1]))
    return point if all(math.isfinite(v) for v in point) else None


def tracking_costs(raw_path, slots, n_frames, pitch, fps, config):
    """Read each tactical row exactly once; missing ball evidence is neutral.

    All frames are validated, including those outside the published half blocks.
    ATD frames are one-based VIDEO frames, not the original EPTS IdFrame counter.
    """
    costs = [array("d"), array("d"), array("d")]
    length, width = pitch
    anchor = None
    stationary = 0
    seen_ball = 0
    threshold = int(math.ceil(config.stationary_seconds * fps))
    count = 0
    with open(raw_path, encoding="utf-8-sig") as handle:
        for count, line in enumerate(handle, 1):
            fields = line.strip().split(":")
            if len(fields) != 3 or int(fields[0]) != count or count > n_frames:
                raise ValueError("Invalid or out-of-order ATD row {}".format(count))
            players = fields[1].split(";")
            if len(players) != len(slots):
                raise ValueError("ATD row {} has {} slots; expected {}".format(
                    count, len(players), len(slots)))
            ball = xy(fields[2])
            nearest = [math.inf, math.inf]
            # Parse players even when the ball is missing to catch malformed input.
            for team, group in zip(slots, players):
                player = xy(group)
                if player is not None and ball is not None:
                    distance = math.hypot((player[0] - ball[0]) * length,
                                          (player[1] - ball[1]) * width)
                    nearest[team] = min(nearest[team], distance)
            evidence = [0.0, 0.0, 0.0]
            if ball is None:
                anchor, stationary = None, 0
            else:
                seen_ball += 1
                bx, by = ball[0] * length, ball[1] * width
                outside = (bx < -config.outside_margin or bx > length + config.outside_margin
                           or by < -config.outside_margin or by > width + config.outside_margin)
                if anchor is None or math.hypot(bx - anchor[0], by - anchor[1]) > config.stationary_radius:
                    anchor, stationary = (bx, by), 1
                else:
                    stationary += 1
                best = min(nearest)
                if outside:
                    # Pitch geometry is positive dead-ball evidence; disappearance is not.
                    evidence = [4.0, 4.0, 0.0]
                elif best <= config.control_radius:
                    # Relative proximity only matters when someone can plausibly control it.
                    evidence = [min(4.0, (d - best) / config.distance_scale)
                                for d in nearest] + [1.5]
                if stationary >= threshold:
                    # Weak evidence: a stationary visible ball can also be a goalkeeper holding it.
                    evidence[0] += 0.75
                    evidence[1] += 0.75
                    evidence[2] = max(0.0, evidence[2] - 0.75)
            for state, cost in enumerate(evidence):
                costs[state].append(cost)
    if count != n_frames:
        raise ValueError("ATD has {} rows; expected {}".format(count, n_frames))
    return costs, {"tracking_rows": count, "ball_observed_frames": seen_ball}


def apply_tags(costs, path, fps, weight):
    """Add capped weak team evidence; overlapping tags cannot multiply confidence.

    Club spellings follow DATA.md, including the export's 'Atletic Club'. Set-piece
    windows surround a restart and are deliberately not treated as dead-ball intervals.
    """
    n = len(costs[0])
    votes = [array("i", [0]) * (n + 1), array("i", [0]) * (n + 1)]
    names = {"barcelona": 0, "fc barcelona": 0, "atletic club": 1, "athletic club": 1}
    used = 0
    for instance in ET.parse(path).iter("instance"):
        code = (instance.findtext("code") or "").strip().casefold()
        if code not in ATTACK | DEFEND:
            continue
        tags = {(label.findtext("group") or "").strip().casefold():
                (label.findtext("text") or "").strip().casefold()
                for label in instance.findall("label")}
        team = names.get(tags.get("team"))
        if team is None:
            continue
        if code in DEFEND:
            team = 1 - team
        start_s, end_s = float(instance.findtext("start")), float(instance.findtext("end"))
        if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s < start_s:
            raise ValueError("Invalid Smart Tagging interval")
        start, end = max(0, min(n, round(start_s * fps))), max(0, min(n, round(end_s * fps)))
        if start < end:
            votes[team][start] += 1
            votes[team][end] -= 1
            used += 1
    active = [0, 0]
    for frame in range(n):
        for team in (0, 1):
            active[team] += votes[team][frame]
        if bool(active[0]) != bool(active[1]):
            owner = 0 if active[0] else 1
            costs[1 - owner][frame] += weight
            costs[2][frame] += weight
    return used


def decode(costs, blocks, config):
    """Exact 3-state Viterbi path, reset between halves. Ties prefer staying.

    Offline inference can use observations on both sides of an occlusion. State
    persistence penalises brief team flips while allowing sustained handovers.
    """
    if len(costs) != 3:
        raise ValueError("Expected three equally sized emission arrays")
    n = len(costs[0])
    if any(len(column) != n for column in costs):
        raise ValueError("Expected three equally sized emission arrays")
    if any(not math.isfinite(c) for column in costs for c in column):
        raise ValueError("Emission costs must be finite")
    labels = ["X"] * n
    last_end = -1
    for _, first, last in blocks:
        if first < 0 or last >= n or first > last or first <= last_end:
            raise ValueError("Half blocks must be ordered, nonoverlapping and in range")
        last_end = last
        previous = [costs[s][first] for s in range(3)]
        parents = bytearray((last - first + 1) * 3)
        for frame in range(first + 1, last + 1):
            current = []
            for state in range(3):
                best, parent = previous[state], state
                for other in range(3):
                    if other == state:
                        continue
                    penalty = config.stoppage_cost if 2 in (state, other) else config.switch_cost
                    candidate = previous[other] + penalty
                    if candidate < best:
                        best, parent = candidate, other
                current.append(best + costs[state][frame])
                parents[(frame - first) * 3 + state] = parent
            # Normalise to keep cumulative costs bounded over long recordings.
            offset = min(current)
            previous = [value - offset for value in current]
        state = min(range(3), key=lambda s: previous[s])
        for frame in range(last, first - 1, -1):
            labels[frame] = STATES[state]
            state = parents[(frame - first) * 3 + state]
    return labels


def write_csv(path, labels):
    if not labels or any(label not in STATES + ("X",) for label in labels):
        raise ValueError("Expected a nonempty H/A/D/X label sequence")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("frame", "label"))
        writer.writerows(enumerate(labels))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", help="Root containing tactical/, or MTAH_DATA_DIR")
    parser.add_argument("--out", default="evaluation/results/sequence.csv")
    parser.add_argument("--no-tags", action="store_true", help="Run a tracking-only ablation")
    args = parser.parse_args()
    config = Config()
    paths = asset_paths(data_dir=args.data_dir)
    required = ["atd_meta", "atd_raw"] + ([] if args.no_tags else ["smart_tagging"])
    missing = [paths[name] for name in required if not Path(paths[name]).is_file()]
    if missing:
        parser.error("Missing tactical assets:\n  " + "\n  ".join(missing))
    output = Path(args.out).resolve()
    if output in {Path(paths[name]).resolve() for name in paths}:
        parser.error("Output must not overwrite a match asset")
    game = GAMES[GAME]
    costs, diagnostics = tracking_costs(paths["atd_raw"], metadata(paths["atd_meta"]),
                                        game["n_video_frames"], game["pitch"], game["fps"], config)
    diagnostics["tactical_tags_used"] = 0 if args.no_tags else apply_tags(
        costs, paths["smart_tagging"], game["fps"], config.tag_weight)
    labels = decode(costs, half_video_blocks(GAME), config)
    write_csv(output, labels)
    diagnostics.update({"game": GAME, "config": asdict(config), "uses_tags": not args.no_tags,
                        "label_counts": {s: labels.count(s) for s in STATES + ("X",)},
                        "input_sha256": {name: sha256(paths[name]) for name in required},
                        "output_sha256": sha256(output)})
    report = output.with_suffix(output.suffix + ".json")
    report.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Wrote {} rows to {}".format(len(labels), output))
    print("Diagnostics: {}".format(report))


if __name__ == "__main__":
    main()
