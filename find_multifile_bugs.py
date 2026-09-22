"""
Find SWE-bench Verified bugs whose REAL fix touches several files.

Why: the pilot sample is 7 single-file bugs and 1 four-file bug. On a single-file bug,
MAP and MFR carry almost no extra information over acc@k, so the interesting metrics are
being decided by one instance. Multi-file bugs are also where localization is structurally
weakest, since `--top_n 3` cannot cover a fix that spans 4 files.

Reads the dataset only. No model calls, nothing run.

Usage:
    python find_multifile_bugs.py                      # distribution + candidates
    python find_multifile_bugs.py --min-files 3        # only 3+ file fixes
    python find_multifile_bugs.py --limit 20
    python find_multifile_bugs.py --categories ../capstone/swebench_bugtypes.csv

Passing --categories cross-references the project's CRASH/LOGIC labels, so candidates can
be chosen to fix the sample's CRASH skew (currently 6 CRASH / 2 LOGIC) at the same time as
its single-file skew.
"""
import argparse
import csv
import os
import re
from collections import Counter, defaultdict

from datasets import load_dataset

FILE_RE = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)

# test files are not localization targets -- the fix we care about is source
TEST_HINTS = ("test_", "/tests/", "_test.py", "/testing/", "conftest.py")


def is_test(path):
    p = path.lower()
    return any(h in p for h in TEST_HINTS)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    parser.add_argument("--min-files", type=int, default=2,
                        help="minimum SOURCE files in the gold patch")
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--categories", default=None,
                        help="optional CSV with instance_id + category columns")
    parser.add_argument("--exclude-used", action="store_true", default=True,
                        help="skip bugs already in the pilot sample")
    args = parser.parse_args()

    already_used = {
        "django__django-15629", "django__django-12262", "django__django-14672",
        "scikit-learn__scikit-learn-10908", "scikit-learn__scikit-learn-25973",
        "pytest-dev__pytest-6202", "django__django-15732", "django__django-13401",
    }

    categories = {}
    if args.categories and os.path.isfile(args.categories):
        with open(args.categories, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                iid = row.get("instance_id")
                cat = row.get("category") or row.get("label") or row.get("bug_type")
                if iid and cat:
                    categories[iid] = cat

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="test")

    dist = Counter()
    by_count = defaultdict(list)

    for row in ds:
        iid = row["instance_id"]
        files = [f for f in FILE_RE.findall(row["patch"] or "") if not is_test(f)]
        n = len(files)
        dist[n] += 1
        if n >= args.min_files:
            by_count[n].append((iid, files))

    total = sum(dist.values())
    print(f"\n=== Source files changed per fix, across all {total} bugs ===")
    for n in sorted(dist):
        bar = "#" * min(60, dist[n] * 60 // max(dist.values()))
        print(f"  {n:>2} file(s): {dist[n]:>4}  {bar}")

    multi = sum(c for n, c in dist.items() if n >= 2)
    print(f"\n  multi-file (2+): {multi}/{total} ({multi / total * 100:.1f}%)")

    print(f"\n=== Candidates with >= {args.min_files} source files ===")
    if categories:
        print(f"{'instance_id':<42} {'files':>5}  {'category':<9} paths")
    else:
        print(f"{'instance_id':<42} {'files':>5}  paths")

    shown = 0
    for n in sorted(by_count, reverse=True):
        for iid, files in sorted(by_count[n]):
            if args.exclude_used and iid in already_used:
                continue
            if shown >= args.limit:
                break
            paths = ", ".join(os.path.basename(f) for f in files[:4])
            if len(files) > 4:
                paths += f", +{len(files) - 4} more"
            if categories:
                print(f"{iid:<42} {n:>5}  {categories.get(iid,'?'):<9} {paths}")
            else:
                print(f"{iid:<42} {n:>5}  {paths}")
            shown += 1
        if shown >= args.limit:
            break

    if categories:
        print("\n=== Category split among the candidates shown ===")
        cc = Counter(categories.get(iid, "?")
                     for n in by_count for iid, _ in by_count[n]
                     if not (args.exclude_used and iid in already_used))
        for cat, count in cc.most_common():
            print(f"  {cat:<10} {count}")
        print("\n  Pilot sample is currently 6 CRASH / 2 LOGIC against an experiment set")
        print("  balanced 50/50 - picking LOGIC multi-file bugs fixes both skews at once.")


if __name__ == "__main__":
    main()
