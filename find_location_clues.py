"""
Does a bug's issue text give away WHERE the fix goes?

For each bug, compares the SWE-bench issue text (problem_statement) with the real fix
(gold patch) and reports every location clue found in the issue:
  FILE      the fixed file's path            e.g. django/db/models/fields/__init__.py
  MODULE    the same file as a module name   e.g. django.db.models.fields
  FILENAME  the file's name alone            e.g. schema.py  (skipped for __init__.py)
  CODE      a function/class the fix edits   e.g. __eq__, Field   (from the patch's
            hunk headers and changed def/class lines)
Each clue is printed with the issue line it appears in, so it can be checked by eye.
Categories loosely follow the "location information" dimension of Xia et al. (2025,
Agentless, Sec. 6.1) - exact location / keywords / not provided - but this is a simple
automatic text match, not their manual classification.

Usage (agentless env, from the Agentless folder):
    python find_location_clues.py                      # the 16 pilot + multi-file bugs
    python find_location_clues.py django__django-13401 # one or more bugs, in detail
"""
import argparse
import re

from datasets import load_dataset

from score_function_localization import gold_functions, make_reader, realign_patch

DEFAULT_BUGS = [
    "django__django-12262", "django__django-13401", "django__django-14672",
    "django__django-15629", "django__django-15732", "pytest-dev__pytest-6202",
    "scikit-learn__scikit-learn-10908", "scikit-learn__scikit-learn-25973",
    "django__django-10554", "django__django-11138", "django__django-11400",
    "django__django-11532", "django__django-11734", "django__django-13121",
    "django__django-13195", "django__django-13344",
]
TOO_GENERIC = {"__init__", "setUp", "test", "main", "run", "get", "set", "__str__"}


def fix_locations(row):
    """Files the fix changes, and the function/class names it edits.

    Functions come from the same gold-function mapping the function-level scorer uses
    (edited lines mapped onto the BUGGY source, after realigning hunk offsets), so a fix
    that edits only the BODY of __eq__ still counts __eq__ (and its class, Field)."""
    patch = row["patch"]
    files = re.findall(r"^diff --git a/\S+ b/(\S+)", patch, re.M)
    read = make_reader(row["repo"], row["base_commit"])
    patch, _ = realign_patch(patch, read)
    names = set()
    for _, func in gold_functions(patch, read):
        for part in (func or "").split("."):                         # Field.__eq__ -> both
            names.add(part)
    for m in re.finditer(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+(\w+)", patch, re.M):
        names.add(m.group(1))                                        # added/removed defs
    return files, sorted(n for n in names if n not in TOO_GENERIC and len(n) > 2)


def clues(issue, files, names):
    found = []
    lines = issue.splitlines()

    def where(pattern):
        return [l.strip() for l in lines if re.search(pattern, l)]

    for f in files:
        mod = f[:-3].replace("/", ".") if f.endswith(".py") else None
        if mod and mod.endswith(".__init__"):
            mod = mod[: -len(".__init__")]
        base = f.rsplit("/", 1)[-1]
        for kind, text in [("FILE", f), ("MODULE", mod),
                           ("FILENAME", None if base == "__init__.py" else base)]:
            if text:
                hits = where(re.escape(text))
                if hits:
                    found.append((kind, text, hits))
    for n in names:
        hits = where(r"\b" + re.escape(n) + r"\b")
        if hits:
            found.append(("CODE", n, hits))
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bugs", nargs="*")
    args = ap.parse_args()
    bugs = args.bugs or DEFAULT_BUGS
    detail = bool(args.bugs)

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    rows = {r["instance_id"]: r for r in ds if r["instance_id"] in set(bugs)}

    summary = []
    for b in bugs:
        r = rows[b]
        files, names = fix_locations(r)
        found = clues(r["problem_statement"], files, names)
        kinds = sorted({k for k, _, _ in found})
        level = ("exact file" if {"FILE", "MODULE"} & set(kinds) else
                 "file name" if "FILENAME" in kinds else
                 "code name" if "CODE" in kinds else "none")
        summary.append((b, level, kinds))
        if detail:
            print(f"\n=== {b}")
            print(f"Real fix changes files : {files}")
            print(f"  ... and functions/classes: {names}")
            if not found:
                print("No location clue found in the issue text.")
            for kind, text, hits in found:
                print(f"\n  [{kind}] '{text}' appears in:")
                for h in hits[:3]:
                    print(f"      > {h[:160]}")

    print(f"\n{'bug':<34}{'strongest clue in issue':<26}found")
    for b, level, kinds in summary:
        print(f"{b:<34}{level:<26}{', '.join(kinds) or '-'}")
    n = len(summary)
    for lvl in ["exact file", "file name", "code name", "none"]:
        k = sum(1 for _, l, _ in summary if l == lvl)
        print(f"  {lvl:<12}: {k}/{n}")


if __name__ == "__main__":
    main()
