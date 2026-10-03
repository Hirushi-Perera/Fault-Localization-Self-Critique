"""
Find BugsInPy bugs whose REAL fix touches several source files.

Why: the only real-bug SBFL run so far (PySnooper bug 3) has one gold file, so it cannot
show whether SBFL finds a fix that is spread across files. SBFL itself needs no change for
this -- it already ranks functions from every source file in one list. What a multi-file
bug tests is whether the failing test actually EXECUTES code in every gold file.

Reads BugsInPy's metadata only (bug.info, bug_patch.txt, requirements.txt). Runs nothing,
installs nothing, no model calls.

Usage:
    python find_bugsinpy_multifile.py
    python find_bugsinpy_multifile.py --min-files 3 --limit 30
    python find_bugsinpy_multifile.py --bugsinpy "C:/path/to/BugsInPy"

Candidates are sorted easiest-to-set-up first: projects without heavy/compiled
dependencies, then fewest pinned requirements, then smallest patch.
"""
import argparse
import os
import re
from collections import Counter, defaultdict

from agentless.critique.sbfl import _is_project_source

DEFAULT_ROOT = "C:/Users/USER/Documents/IIT/4thYear/FYP/BugsInPy"
FILE_RE = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)

# Projects whose environments are large or need compiled extensions (numpy/C/Cython,
# TensorFlow, etc.). Not impossible without Docker, just the most likely to fail on a
# Windows laptop -- this is a judgement call, not a measured fact.
HEAVY = {"keras", "pandas", "matplotlib", "spacy", "scrapy", "ansible", "luigi"}


def read_info(bug_dir):
    info = {}
    with open(os.path.join(bug_dir, "bug.info"), encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                info[k.strip()] = v.strip().strip('"')
    return info


def count_requirements(bug_dir):
    path = os.path.join(bug_dir, "requirements.txt")
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8", errors="replace") as f:
        return sum(1 for l in f if l.strip() and not l.lstrip().startswith("#"))


def patch_stats(bug_dir):
    """(source files changed, number of changed +/- lines in those files)."""
    with open(os.path.join(bug_dir, "bug_patch.txt"), encoding="utf-8", errors="replace") as f:
        patch = f.read()
    files, changed, current = [], 0, None
    for line in patch.splitlines():
        m = FILE_RE.match(line)
        if m:
            current = m.group(1) if _is_project_source(m.group(1)) else None
            if current and current not in files:
                files.append(current)
            continue
        if current and line[:1] in "+-" and not line.startswith(("+++", "---")):
            changed += 1
    return files, changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bugsinpy", default=DEFAULT_ROOT)
    ap.add_argument("--min-files", type=int, default=2)
    ap.add_argument("--limit", type=int, default=20)
    args = ap.parse_args()

    projects_dir = os.path.join(args.bugsinpy, "projects")
    rows, per_project, total_dist = [], defaultdict(Counter), Counter()

    for project in sorted(os.listdir(projects_dir)):
        bugs_dir = os.path.join(projects_dir, project, "bugs")
        if not os.path.isdir(bugs_dir):
            continue
        for bug in sorted(os.listdir(bugs_dir), key=lambda b: int(b) if b.isdigit() else 0):
            bug_dir = os.path.join(bugs_dir, bug)
            if not os.path.exists(os.path.join(bug_dir, "bug_patch.txt")):
                continue
            files, changed = patch_stats(bug_dir)
            n = len(files)
            total_dist[n] += 1
            per_project[project]["multi" if n >= 2 else "single"] += 1
            if n >= args.min_files:
                info = read_info(bug_dir)
                rows.append({
                    "project": project, "bug": bug, "files": files, "changed": changed,
                    "python": info.get("python_version", "?"),
                    "tests": info.get("test_file", "").split(";"),
                    "reqs": count_requirements(bug_dir),
                    "heavy": project in HEAVY,
                })

    total = sum(total_dist.values())
    print(f"BugsInPy bugs with a patch: {total}")
    print("Source files changed by the fix:")
    for n in sorted(total_dist):
        print(f"  {n} file(s): {total_dist[n]:4d}  ({100 * total_dist[n] / total:.1f}%)")

    print("\nMulti-file bugs per project (multi / total):")
    for p in sorted(per_project):
        c = per_project[p]
        tag = "  [heavy deps]" if p in HEAVY else ""
        print(f"  {p:14s} {c['multi']:3d} / {c['multi'] + c['single']:3d}{tag}")

    rows.sort(key=lambda r: (r["heavy"], r["reqs"], r["changed"]))
    print(f"\nCandidates with >= {args.min_files} gold files, easiest setup first "
          f"(showing {min(args.limit, len(rows))} of {len(rows)}):")
    for r in rows[:args.limit]:
        heavy = " HEAVY" if r["heavy"] else ""
        print(f"\n  {r['project']} bug {r['bug']}  |  {len(r['files'])} files, "
              f"{r['changed']} changed lines, python {r['python']}, "
              f"{r['reqs']} pinned reqs{heavy}")
        for f in r["files"]:
            print(f"      gold: {f}")
        print(f"      test: {', '.join(t for t in r['tests'] if t)}")


if __name__ == "__main__":
    main()
