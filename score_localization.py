"""
Score file-level localization with the standard IR / bug-localization metrics.

Re-scores localization runs that have ALREADY happened -- reads the existing
results/pilot_rerun/<bug>/loc_outputs.jsonl files and compares them against the
real developer patch from SWE-bench Verified. No model calls, no re-running the
pipeline: same predictions, just measured properly.

Usage:
    python score_localization.py
    python score_localization.py --results_dir results/pilot_rerun

Metrics (see the calculation notes on each function below):
    Top-1 / Top-3   did a gold file appear in the first 1 / first 3 guesses?
    MRR             how HIGH up was the first correct guess?
    MAP             did we find ALL the gold files, not just one?
"""
import argparse
import json
import os
import re

from datasets import load_dataset

GOLD_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def normalise(path):
    """Make two paths comparable: forward slashes, no './' prefix, lowercase."""
    return (path or "").strip().replace("\\", "/").lstrip("./").lower()


def gold_files_from_patch(patch):
    """Every file the real developer's fix touched, read off the diff headers."""
    return {normalise(m.group(1)) for m in GOLD_FILE_RE.finditer(patch or "")}


# --------------------------------------------------------------------------
# THE METRICS
#
# Throughout: `predicted` is the RANKED list of files the model guessed
# (best guess first), `gold` is the set of files the real fix actually changed.
# --------------------------------------------------------------------------

def top_n(predicted, gold, n):
    """
    Top-N accuracy: 1 if ANY gold file is in the first n guesses, else 0.

    The blunt one. Answers "did we find it at all?" and nothing more --
    ignores both where it ranked and how many gold files we missed.
    """
    return 1 if any(normalise(p) in gold for p in predicted[:n]) else 0


def reciprocal_rank(predicted, gold):
    """
    Reciprocal Rank: 1 / (position of the FIRST correct guess). 0 if none.

        gold file guessed 1st -> 1/1 = 1.00
        gold file guessed 2nd -> 1/2 = 0.50
        gold file guessed 3rd -> 1/3 = 0.33

    Rewards ranking the right file HIGH, not merely including it somewhere.
    Averaged over all bugs, this is MRR.
    """
    for i, p in enumerate(predicted, start=1):
        if normalise(p) in gold:
            return 1.0 / i
    return 0.0


def average_precision(predicted, gold):
    """
    Average Precision: rewards finding ALL the gold files, not just one.

    Walk down the ranked list. Every time we hit a gold file, record the
    precision at that point (how many of the guesses so far were correct).
    Then average those, dividing by the TOTAL number of gold files -- which is
    what penalises misses.

    Worked example: gold = 4 files, we guessed 1 correctly, at position 1.
        hit at position 1 -> precision = 1 correct / 1 guess = 1.00
        AP = 1.00 / 4 gold files = 0.25
    Top-3 calls that a full hit (1.0). AP calls it 0.25. That gap is the
    whole reason this metric is here.

    Averaged over all bugs, this is MAP.
    """
    if not gold:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, p in enumerate(predicted, start=1):
        if normalise(p) in gold:
            hits += 1
            precision_sum += hits / i          # precision@i, at a hit
    return precision_sum / len(gold)           # divide by ALL gold files


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/pilot_rerun")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="test")
    patches = {r["instance_id"]: r["patch"] for r in ds}

    bugs = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isfile(os.path.join(args.results_dir, d, "loc_outputs.jsonl"))
    )

    rows = []
    for bug in bugs:
        loc_path = os.path.join(args.results_dir, bug, "loc_outputs.jsonl")
        with open(loc_path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        predicted = record.get("found_files") or []
        gold = gold_files_from_patch(patches.get(bug))

        if not gold:
            print(f"  ! no gold patch found for {bug}, skipping")
            continue

        found = sum(1 for g in gold if g in {normalise(p) for p in predicted})
        rows.append({
            "bug": bug,
            "n_pred": len(predicted),
            "n_gold": len(gold),
            "found": found,
            "top1": top_n(predicted, gold, 1),
            "top3": top_n(predicted, gold, 3),
            "rr": reciprocal_rank(predicted, gold),
            "ap": average_precision(predicted, gold),
        })

    if not rows:
        print("No scored bugs -- check --results_dir")
        return

    header = (f"{'bug':<34} {'pred':>4} {'gold':>4} {'found':>5} "
              f"{'Top-1':>5} {'Top-3':>5} {'RR':>5} {'AP':>5}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['bug']:<34} {r['n_pred']:>4} {r['n_gold']:>4} {r['found']:>5} "
              f"{r['top1']:>5} {r['top3']:>5} {r['rr']:>5.2f} {r['ap']:>5.2f}")
    print("-" * len(header))

    n = len(rows)
    print(f"\nn = {n} bugs")
    print(f"  Top-1 accuracy : {sum(r['top1'] for r in rows) / n * 100:5.1f} %"
          f"   ({sum(r['top1'] for r in rows)}/{n})")
    print(f"  Top-3 accuracy : {sum(r['top3'] for r in rows) / n * 100:5.1f} %"
          f"   ({sum(r['top3'] for r in rows)}/{n})")
    print(f"  MRR            : {sum(r['rr'] for r in rows) / n:5.3f}")
    print(f"  MAP            : {sum(r['ap'] for r in rows) / n:5.3f}")

    # ---- precision / recall / F1, pooled over every file across every bug ----
    # recall    = of all the files the real fixes changed, how many did we find?
    # precision = of all the files we guessed, how many were actually right?
    #             Low by design: we return 3-5 candidates when most bugs need 1.
    total_gold = sum(r["n_gold"] for r in rows)
    total_found = sum(r["found"] for r in rows)
    total_pred = sum(r["n_pred"] for r in rows)
    recall = total_found / total_gold if total_gold else 0.0
    precision = total_found / total_pred if total_pred else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    print(f"  Recall         : {recall * 100:5.1f} %   "
          f"({total_found}/{total_gold} gold files found)")
    print(f"  Precision      : {precision * 100:5.1f} %   "
          f"({total_found}/{total_pred} guesses correct)")
    print(f"  F1             : {f1:5.3f}")

    # ---- single-file vs multi-file split -------------------------------------
    # Matters because a 1-gold-file bug makes AP and RR identical by definition,
    # so a sample dominated by single-file fixes makes MAP look better than it is.
    single = [r for r in rows if r["n_gold"] == 1]
    multi = [r for r in rows if r["n_gold"] > 1]
    print("\n  Breakdown by fix size:")
    for label, group in (("single-file", single), ("multi-file ", multi)):
        if not group:
            print(f"    {label}: none in this sample")
            continue
        g = len(group)
        print(f"    {label} (n={g}): "
              f"Top-1 {sum(r['top1'] for r in group) / g * 100:.0f}%  "
              f"Top-3 {sum(r['top3'] for r in group) / g * 100:.0f}%  "
              f"MRR {sum(r['rr'] for r in group) / g:.3f}  "
              f"MAP {sum(r['ap'] for r in group) / g:.3f}")


if __name__ == "__main__":
    main()
