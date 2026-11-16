"""Profile tracking and stress-test a frozen predictor without opening any labels.

Outputs are aggregate stability diagnostics, never football accuracy estimates.
Temporary corrupted tracking stays under the gitignored output directory.
"""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import tempfile
import time

from games import GAME, GAMES, asset_paths, half_video_blocks
from solution.predict import Config, apply_tags, decode, metadata, sha256, tracking_costs, xy


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def rows(path, n_frames, n_slots):
    count = 0
    with open(path, encoding="utf-8-sig") as handle:
        for count, line in enumerate(handle, 1):
            fields = line.rstrip("\r\n").split(":")
            if len(fields) != 3 or int(fields[0]) != count or count > n_frames:
                raise ValueError("Invalid or out-of-order tracking row {}".format(count))
            players = fields[1].split(";")
            if len(players) != n_slots:
                raise ValueError("Wrong slot count at row {}".format(count))
            # Validate all coordinates, including frames with a missing ball.
            for player in players:
                xy(player)
            yield count - 1, fields, xy(fields[2])
    if count != n_frames:
        raise ValueError("Expected {} tracking rows; got {}".format(n_frames, count))


def run_lengths(values):
    lengths, current = [], 0
    for value in values:
        if value:
            current += 1
        elif current:
            lengths.append(current)
            current = 0
    if current:
        lengths.append(current)
    return lengths


def duration_summary(lengths, fps):
    ordered = sorted(lengths)
    def quantile(q):
        return ordered[max(0, math.ceil(q * len(ordered)) - 1)] / fps if ordered else 0.0
    return {"runs": len(ordered), "total_seconds": sum(ordered) / fps,
            "p50_seconds": quantile(.5), "p90_seconds": quantile(.9),
            "p99_seconds": quantile(.99), "max_seconds": quantile(1)}


def profile(raw, slots, n_frames, blocks, fps):
    observed = bytearray(n_frames)
    for frame, _, ball in rows(raw, n_frames, len(slots)):
        observed[frame] = ball is not None
    halves = []
    for half, first, last in blocks:
        values = observed[first:last + 1]
        halves.append({"half": half, "frames": len(values),
                       "observed_ball_frames": sum(values),
                       "missing_fraction": 1 - sum(values) / len(values),
                       "missing_runs": duration_summary(run_lengths(not v for v in values), fps)})
    return {"frames": n_frames, "observed_ball_frames": sum(observed),
            "halves": halves}, observed


def mask_frames(n_frames, blocks, rate, mode, seed, burst_frames):
    """Timeline mask independent of observations; reset at every half.

    Bursts are a stationary two-state Markov chain: P(drop->keep)=1/L,
    P(keep->drop)=p/((1-p)*L). L is the expected dropped-run duration.
    p describes the marginal TIMELINE mask rate, not an exact visible-ball count.
    """
    if not 0 <= rate < 1 or not math.isfinite(rate):
        raise ValueError("rate must be finite and in [0, 1)")
    if mode not in ("iid", "burst"):
        raise ValueError("Unknown corruption mode")
    if burst_frames < 1 or rate / (1 - rate) / burst_frames > 1:
        raise ValueError("Invalid burst duration for this rate")
    mask = bytearray(n_frames)
    rng = random.Random(seed)
    for _, first, last in blocks:
        if first < 0 or last >= n_frames or first > last:
            raise ValueError("Invalid half block")
        dropped = rng.random() < rate
        for frame in range(first, last + 1):
            if mode == "iid":
                dropped = rng.random() < rate
            elif frame != first:
                if dropped:
                    dropped = not (rng.random() < 1 / burst_frames)
                else:
                    dropped = rng.random() < rate / ((1 - rate) * burst_frames)
            mask[frame] = dropped
    return mask


def corrupted_copy(raw, target, mask, n_frames, n_slots):
    removed = 0
    with open(target, "w", encoding="utf-8", newline="") as handle:
        for frame, fields, ball in rows(raw, n_frames, n_slots):
            if mask[frame] and ball is not None:
                fields[2] = "NaN,NaN,NaN"
                removed += 1
            handle.write(":".join(fields) + "\n")
    return removed


def predict(raw, paths, slots, game, blocks, config):
    costs, _ = tracking_costs(raw, slots, game["n_video_frames"], game["pitch"],
                              game["fps"], config)
    apply_tags(costs, paths["smart_tagging"], game["fps"], config.tag_weight)
    return decode(costs, blocks, config)


def sequence_summary(labels, blocks, fps):
    counts, possessions, transitions = Counter(), [], 0
    for _, first, last in blocks:
        values = labels[first:last + 1]
        counts.update(values)
        transitions += sum(a != b for a, b in zip(values, values[1:]))
        start = 0
        for i in range(1, len(values) + 1):
            if i == len(values) or values[i] != values[start]:
                if values[start] in ("H", "A"):
                    possessions.append(i - start)
                start = i
    return {"label_counts_in_halves": dict(counts), "state_transitions": transitions,
            "possession_durations": duration_summary(possessions, fps)}


def agreement(clean, corrupted, blocks, masked):
    totals, flips = Counter(), Counter()
    for _, first, last in blocks:
        for frame in range(first, last + 1):
            group = "masked" if masked[frame] else "unmasked"
            for key in ("all", group):
                totals[key] += 1
                flips[key] += clean[frame] != corrupted[frame]
    return {key: {"frames": totals[key], "changed_frames": flips[key],
                  "changed_fraction": flips[key] / totals[key] if totals[key] else None}
            for key in ("all", "masked", "unmasked")}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir")
    parser.add_argument("--out", required=True, help="New directory under research/results/")
    parser.add_argument("--profile-only", action="store_true")
    parser.add_argument("--rates", type=float, nargs="+", default=[.1, .3, .5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    parser.add_argument("--modes", choices=["iid", "burst"], nargs="+", default=["iid", "burst"])
    parser.add_argument("--burst-seconds", type=float, default=2.0)
    args = parser.parse_args()
    game, blocks, config = GAMES[GAME], half_video_blocks(GAME), Config()
    n, fps = game["n_video_frames"], game["fps"]
    # Validate before creating outputs or reading large assets.
    if not math.isfinite(args.burst_seconds) or args.burst_seconds * fps < 1:
        parser.error("burst-seconds must be finite and at least one frame")
    for rate in args.rates:
        mask_frames(1, [(1, 0, 0)], rate, "burst", 0, args.burst_seconds * fps)
    out = Path(args.out).resolve()
    allowed = Path("research/results").resolve()
    if allowed not in out.parents:
        parser.error("output must be a new subdirectory of research/results/ (gitignored)")
    out.mkdir(parents=True, exist_ok=False)
    paths = asset_paths(data_dir=args.data_dir)
    slots = metadata(paths["atd_meta"])
    started = time.monotonic()
    tracking_profile, observed = profile(paths["atd_raw"], slots, n, blocks, fps)
    save_json(out / "tracking_profile.json", tracking_profile)
    source_paths = ["games.py", "solution/predict.py", "research/robustness.py"]
    report = {"schema_version": 1, "interpretation": "stability_only_not_accuracy",
              "game": GAME, "config": asdict(config), "uses_tags": True,
              "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "python": platform.python_version(),
              "source_sha256": {p: sha256(p) for p in source_paths},
              "input_sha256": {k: sha256(paths[k]) for k in ("atd_raw", "atd_meta", "smart_tagging")},
              "design": {"rates": args.rates, "seeds": args.seeds, "modes": args.modes,
                         "burst_seconds": args.burst_seconds,
                         "mask_rate_definition": "stationary marginal over half timeline"},
              "complete": False, "runs": []}
    save_json(out / "summary.json", report)
    if not args.profile_only:
        clean = predict(paths["atd_raw"], paths, slots, game, blocks, config)
        report["clean_prediction_label_sha256"] = hashlib.sha256("".join(clean).encode()).hexdigest()
        report["clean"] = sequence_summary(clean, blocks, fps)
        eligible_observed = sum(sum(observed[first:last + 1]) for _, first, last in blocks)
        for mode in args.modes:
            for rate in args.rates:
                for seed in args.seeds:
                    mask = mask_frames(n, blocks, rate, mode, seed, args.burst_seconds * fps)
                    # Never persist or publish reconstructive tracking exports.
                    with tempfile.TemporaryDirectory(prefix="private-", dir=out) as temporary:
                        raw = Path(temporary) / "corrupted.txt"
                        removed = corrupted_copy(paths["atd_raw"], raw, mask, n, len(slots))
                        labels = predict(raw, paths, slots, game, blocks, config)
                    record = {"mode": mode, "rate": rate, "seed": seed,
                              "mask_sha256": hashlib.sha256(mask).hexdigest(),
                              "removed_observed_ball_frames": removed,
                              "removed_fraction_of_original_observed_in_halves":
                                  removed / eligible_observed if eligible_observed else None,
                              "agreement_to_clean": agreement(clean, labels, blocks, mask),
                              "sequence": sequence_summary(labels, blocks, fps)}
                    report["runs"].append(record)
                    save_json(out / "summary.json", report)
                    print("Completed {} rate={} seed={}".format(mode, rate, seed), flush=True)
    report["complete"] = True
    save_json(out / "summary.json", report)
    save_json(out / "runtime.json", {"wall_seconds": time.monotonic() - started})
    print("Aggregate diagnostics: {}".format(out), flush=True)


if __name__ == "__main__":
    main()
