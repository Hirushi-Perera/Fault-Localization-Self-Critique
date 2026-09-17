"""
Score the repair stage: how often does the pipeline produce a patch that APPLIES,
and does it touch a file the real developer's fix touched?

Reads the already-written repair outputs -- no model calls, no re-running anything.

Usage:
    python score_repair.py
    python score_repair.py --results_dir results/pilot_full

IMPORTANT about what this does and does not measure:
    "applies"      = the SEARCH/REPLACE edit matched real file content and produced a
                     non-empty diff. This is what we measure here.
    "correct"      = the patch actually fixes the bug. This needs SWE-bench's own test
                     suite (Docker), which is NOT run here. Do not report the numbers
                     below as a fix rate.
"""
import argparse
import json
import os
import re

from datasets import load_dataset

GOLD_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)
PATCH_FILE_RE = re.compile(r"^diff --git a/(\S+) b/(\S+)", re.MULTILINE)


def normalise(path):
    return (path or "").strip().replace("\\", "/").lstrip("./").lower()


def files_in_patch(patch):
    return {normalise(m.group(1)) for m in PATCH_FILE_RE.finditer(patch or "")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/pilot_full")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="test")
    gold = {r["instance_id"]: r["patch"] for r in ds}

    bugs = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isdir(os.path.join(args.results_dir, d))
    )

    rows = []
    for bug in bugs:
        processed = os.path.join(
            args.results_dir, bug, "repair", "output_0_processed.jsonl"
        )
        if not os.path.isfile(processed):
            rows.append({"bug": bug, "status": "no repair output",
                         "applied": False, "right_file": None, "lines": 0})
            continue

        with open(processed, encoding="utf-8") as f:
            record = json.loads(f.readline())

        patch = record.get("model_patch") or ""
        gold_files = files_in_patch(gold.get(bug))
        pred_files = files_in_patch(patch)

        if not patch.strip():
            rows.append({"bug": bug, "status": "empty patch",
                         "applied": False, "right_file": None, "lines": 0})
            continue

        overlap = pred_files & gold_files
        # count changed lines as a rough size signal (+/- lines, excluding headers)
        changed = sum(
            1 for ln in patch.splitlines()
            if (ln.startswith("+") or ln.startswith("-"))
            and not ln.startswith(("+++", "---"))
        )
        rows.append({
            "bug": bug,
            "status": "applied",
            "applied": True,
            "right_file": bool(overlap),
            "lines": changed,
        })

    header = f"{'bug':<34} {'status':<16} {'gold file?':<11} {'lines'}"
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        rf = "-" if r["right_file"] is None else ("yes" if r["right_file"] else "NO")
        print(f"{r['bug']:<34} {r['status']:<16} {rf:<11} {r['lines']}")
    print("-" * len(header))

    n = len(rows)
    applied = sum(1 for r in rows if r["applied"])
    right = sum(1 for r in rows if r["right_file"])
    print(f"\nn = {n} bugs")
    print(f"  Patch applies          : {applied}/{n}  ({applied / n * 100:.1f} %)")
    print(f"  ... and touches a gold file: {right}/{n}  ({right / n * 100:.1f} %)")
    print("\n  NOTE: 'applies' is not 'fixes'. Confirming a fix needs SWE-bench's")
    print("  test suite under Docker, which this script does not run.")


if __name__ == "__main__":
    main()
