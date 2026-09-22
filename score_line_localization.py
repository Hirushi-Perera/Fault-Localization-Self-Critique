"""
Score LINE-level localization (Agentless stage 3) against the real developer's patch.

Reads results already on disk -- `<bug>/edit_loc/loc_outputs.jsonl` -- and compares the
line numbers the pipeline predicted against the lines the gold patch actually changed.
No model calls, nothing re-run.

Usage:
    python score_line_localization.py
    python score_line_localization.py --results_dir results/pilot_full

Gold lines come from `faithfulness_sketch.parse_patch_old_lines()` (reused rather than
reimplemented): it walks the diff per hunk, treating removed '-' lines as the precise
location, and falling back to anchors either side of the insertion point for pure-insertion
hunks where nothing was removed.

WHY TOLERANCE BANDS: exact line agreement is a very strict bar, and the pilot already
showed why it matters -- on pytest-6202 the pipeline landed ~18 lines from the real fault,
which was far enough that a +/-10 repair context window never showed the model the buggy
line at all. So the useful question is not only "exact?" but "close enough for the repair
stage to see it?". Both are reported.
"""
import argparse
import json
import os
import re
import sys

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "capstone")))
from faithfulness_sketch import parse_patch_old_lines  # noqa: E402

LINE_RE = re.compile(r"^\s*line:\s*(\d+)", re.MULTILINE)
TOLERANCES = [0, 5, 10, 25]


def normalise(path):
    return (path or "").strip().replace("\\", "/").lstrip("./").lower()


def predicted_lines(record):
    """
    found_edit_locs -> ordered list of (file, line), best guess first.

    Shape is {file: [text, ...]} where each text holds 'function: X' / 'line: N'
    entries; order within the text is the model's own ranking.
    """
    locs = record.get("found_edit_locs") or {}
    if isinstance(locs, list):                      # num_samples>1 shape
        locs = locs[0] if locs else {}
    out = []
    for path, blocks in locs.items():
        if isinstance(blocks, str):
            blocks = [blocks]
        for block in blocks or []:
            for m in LINE_RE.finditer(block or ""):
                out.append((normalise(path), int(m.group(1))))
    return out


def distance_to_gold(pred_file, pred_line, gold):
    """Smallest line distance to a gold-changed line IN THE SAME FILE, else None."""
    best = None
    for gfile, glines in gold.items():
        gf = normalise(gfile)
        if not (gf == pred_file or gf.endswith(pred_file) or pred_file.endswith(gf)):
            continue
        for gl in glines:
            d = abs(gl - pred_line)
            if best is None or d < best:
                best = d
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/pilot_full")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="test")
    patches = {r["instance_id"]: r["patch"] for r in ds}

    bugs = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isfile(os.path.join(args.results_dir, d,
                                       "edit_loc", "loc_outputs.jsonl"))
    )

    rows = []
    for bug in bugs:
        path = os.path.join(args.results_dir, bug, "edit_loc", "loc_outputs.jsonl")
        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        preds = predicted_lines(record)
        gold = parse_patch_old_lines(patches.get(bug) or "")

        # distance of each prediction, in rank order; None = wrong file entirely
        dists = [distance_to_gold(pf, pl, gold) for pf, pl in preds]
        finite = [d for d in dists if d is not None]

        rows.append({
            "bug": bug,
            "n_pred": len(preds),
            "n_gold_lines": sum(len(v) for v in gold.values()),
            "dists": dists,
            "best": min(finite) if finite else None,
            "first": dists[0] if dists else None,
        })

    # ---------------- per-bug table ----------------
    header = (f"{'bug':<34} {'preds':>5} {'gold':>5} {'1st pred':>9} {'closest':>8}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        first = "wrong file" if r["first"] is None else f"{r['first']} lines"
        best = "none" if r["best"] is None else f"{r['best']} lines"
        print(f"{r['bug']:<34} {r['n_pred']:>5} {r['n_gold_lines']:>5} "
              f"{first:>9} {best:>8}")
    print("-" * len(header))

    # ---------------- acc@k at line level ----------------
    n = len(rows) or 1
    print(f"\nn = {len(rows)} bugs\n")
    print("LINE-LEVEL acc@k -- a bug counts as a hit if a predicted line falls")
    print("within the tolerance of a line the real fix changed.\n")
    print(f"  {'tolerance':<14} {'acc@1':>8} {'acc@3':>8} {'acc@any':>8}")
    for tol in TOLERANCES:
        def hit_within(r, k):
            for d in r["dists"][:k] if k else r["dists"]:
                if d is not None and d <= tol:
                    return True
            return False

        a1 = sum(hit_within(r, 1) for r in rows) / n * 100
        a3 = sum(hit_within(r, 3) for r in rows) / n * 100
        aa = sum(hit_within(r, None) for r in rows) / n * 100
        label = "exact" if tol == 0 else f"+/-{tol} lines"
        print(f"  {label:<14} {a1:>7.1f}% {a3:>7.1f}% {aa:>7.1f}%")

    # ---------------- MFR, line level ----------------
    ranks = []
    for r in rows:
        rank = next((i for i, d in enumerate(r["dists"], start=1)
                     if d is not None and d <= 10), None)
        if rank:
            ranks.append(rank)
    if ranks:
        print(f"\n  MFR (first prediction within +/-10 lines): "
              f"{sum(ranks) / len(ranks):.3f}   "
              f"[over the {len(ranks)}/{len(rows)} bugs that had any such hit]")

    print("\n  NOTE: +/-25 matters because repair runs with --context_window 25, so a")
    print("  prediction further off than that never shows the model the buggy line.")


if __name__ == "__main__":
    main()
