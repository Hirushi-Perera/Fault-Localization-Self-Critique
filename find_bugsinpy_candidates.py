"""
Shortlist BugsInPy bugs for the two Agentless-on-BugsInPy batches, mirroring the
SWE-bench runs:

    batch 1 (pilot)       - single-file fixes (the SWE-bench pilot was 7 single + 1 multi)
    batch 2 (multi-file)  - fixes changing 2+ source files

Reads BugsInPy's metadata only (bug.info, bug_patch.txt, requirements.txt, run_test.sh).
Runs nothing. CRASH/LOGIC cannot be known from metadata - it comes from actually running
the triggering test, which bugsinpy_to_agentless.py does. So shortlist MORE than 8 per
batch, convert them, and keep a 4 CRASH / 4 LOGIC set.

Ranking, easiest to run first:
  1. a conda env that already exists for the bug's Python version
  2. not a heavy project (compiled / very large dependencies)
  3. fewest pinned requirements  4. fewest triggering tests  5. smallest patch
At most --per-project bugs from one project, so one project cannot dominate a batch.

Usage:
    python find_bugsinpy_candidates.py
    python find_bugsinpy_candidates.py --per-project 3 --show 15
"""
import argparse
import os
from collections import Counter

from find_bugsinpy_multifile import DEFAULT_ROOT, HEAVY, patch_stats, read_info

# Python version -> the conda env that covers it (None = would need a new env)
ENVS = {"3.8": "bugsinpy38", "3.7": "bugsinpy37"}

# Known run problems on this laptop (from the SBFL work) - shown, not excluded:
# Agentless only needs the triggering test to run once, not the whole suite.
NOTES = {
    "tornado": "full suite hangs on Windows; async tests (fine for Agentless)",
    "PySnooper": "bug 1 needs python-toolbox; bug 3 needs `future`",
}


def count_requirements(bug_dir):
    """Pinned requirements, reading UTF-16 files too (some BugsInPy files are UTF-16)."""
    path = os.path.join(bug_dir, "requirements.txt")
    if not os.path.exists(path):
        return 0
    raw = open(path, "rb").read()
    text = raw.decode("utf-16") if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else \
        raw.decode("utf-8", errors="replace")
    return sum(1 for l in text.splitlines()
               if l.strip() and not l.lstrip().startswith(("#", "-e")))


def count_triggers(bug_dir):
    path = os.path.join(bug_dir, "run_test.sh")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for l in f if l.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugsinpy", default=DEFAULT_ROOT)
    ap.add_argument("--per-project", type=int, default=3)
    ap.add_argument("--show", type=int, default=14, help="candidates listed per batch")
    args = ap.parse_args()

    projects_dir = os.path.join(args.bugsinpy, "projects")
    rows = []
    for project in sorted(os.listdir(projects_dir)):
        bugs_dir = os.path.join(projects_dir, project, "bugs")
        if not os.path.isdir(bugs_dir):
            continue
        for bug in os.listdir(bugs_dir):
            bug_dir = os.path.join(bugs_dir, bug)
            if not os.path.exists(os.path.join(bug_dir, "bug_patch.txt")):
                continue
            files, changed = patch_stats(bug_dir)
            if not files:
                continue                      # fix touches no .py source file
            info = read_info(bug_dir)
            py = info.get("python_version", "?").strip()
            env = ENVS.get(".".join(py.split(".")[:2]))
            rows.append({
                "project": project, "bug": bug, "files": files, "changed": changed,
                "python": py, "env": env or f"NEW env (python {py})",
                "heavy": project in HEAVY, "reqs": count_requirements(bug_dir),
                "triggers": count_triggers(bug_dir),
            })

    def rank(r):
        return (r["env"].startswith("NEW"), r["heavy"], r["reqs"], r["triggers"], r["changed"])

    for title, pick in (("BATCH 1 - single-file fixes", lambda r: len(r["files"]) == 1),
                        ("BATCH 2 - multi-file fixes", lambda r: len(r["files"]) >= 2)):
        pool = sorted([r for r in rows if pick(r)], key=rank)
        per_proj, shown = Counter(), []
        for r in pool:
            if per_proj[r["project"]] < args.per_project:
                per_proj[r["project"]] += 1
                shown.append(r)
            if len(shown) == args.show:
                break
        print(f"\n{'=' * 78}\n{title}: {len(pool)} in BugsInPy, showing {len(shown)} "
              f"(max {args.per_project} per project)\n{'=' * 78}")
        print(f"{'bug':18s} {'files':>5s} {'lines':>5s} {'trig':>4s} {'reqs':>4s}  {'env':12s} gold files")
        for r in shown:
            name = f"{r['project']} {r['bug']}"
            print(f"{name:18s} {len(r['files']):>5d} {r['changed']:>5d} {r['triggers']:>4d} "
                  f"{r['reqs']:>4d}  {r['env']:12s} {', '.join(r['files'])}")
        notes = {r["project"] for r in shown} & set(NOTES)
        for p in sorted(notes):
            print(f"  note - {p}: {NOTES[p]}")

    print("\nNext: run bugsinpy_to_agentless.py on candidates (in the env shown) until each "
          "batch has 4 CRASH + 4 LOGIC; the label is printed as 'Failure types: ... -> X'.")


if __name__ == "__main__":
    main()
