"""
The evidence chain for one bug, straight from the dataset and the run logs:
  1. the real fix (the developer's patch shipped with the dataset) -> the "buggy files"
  2. the stack trace the model was given in the no-issue run (show_traceback.py marks it up)
  3. the files the model picked, with and without the issue text

Usage (agentless env, from the Agentless folder):
    python show_evidence.py django__django-11734
    python show_evidence.py bugsinpy__tornado-9
    python show_evidence.py django__django-11734 --diff    # also print the full fix
"""
import argparse
import json
import os
import re

SOURCES = {   # dataset file -> run folders, issue-text run first
    "datasets/swebench_noissue/test.jsonl": ["results/pilot_rerun", "results/multifile",
                                              "results/swebench_noissue"],
    "datasets/bugsinpy_pilot/test.jsonl": ["results/bugsinpy_pilot"],
    "datasets/bugsinpy_multifile/test.jsonl": ["results/bugsinpy_multifile"],
}


def find(bug):
    for path, runs in SOURCES.items():
        with open(path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r["instance_id"] == bug:
                    return r, runs
    raise SystemExit(f"{bug} not found")


def picks(run, bug):
    p = os.path.join(run, bug, "loc_outputs.jsonl")
    if not os.path.isfile(p):
        return None
    return json.loads(open(p, encoding="utf-8").readline()).get("found_files") or []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bug")
    ap.add_argument("--diff", action="store_true")
    args = ap.parse_args()
    r, runs = find(args.bug)

    gold = re.findall(r"^diff --git a/\S+ b/(\S+)", r["patch"], re.M)
    print(f"=== 1. Real fix (dataset field 'patch'; repo {r.get('repo', '?')}, "
          f"base commit {str(r.get('base_commit', '?'))[:10]})")
    for g in gold:
        print(f"   changed: {g}")
    if args.diff:
        print(r["patch"])

    print("\n=== 2. Trace the model was given (no-issue run): fix-file lines + the error")
    for t in re.findall(r"call stack:\n\n```\n(.*?)\n```", r["problem_statement"], re.S):
        frames = [l for l in t.splitlines() if l.strip().startswith("File ")]
        hits = [l for l in frames if any(f'"{g}"' in l or f'/{g}"' in l for g in gold)]
        print(f"   {len(frames)} frames, {len(hits)} in a fix file")
        for l in hits[-4:]:
            print(f"   {l.strip()[:150]}   <== fix file")
        print(f"   last line: {t.strip().splitlines()[-1][:150]}\n")

    print("=== 3. Files the model picked (first = its top answer)")
    labels = (["with issue text", "with issue text", "without issue text"]
              if len(runs) == 3 else ["failing-test report"])
    for run, label in zip(runs, labels):
        p = picks(run, args.bug)
        if p is None:
            continue
        marks = [f"{f}{' *' if f in gold else ''}" for f in p[:3]]
        print(f"   {label:<20} ({run}): {marks}")
    print("   (* = a file the real fix changed)")


if __name__ == "__main__":
    main()
