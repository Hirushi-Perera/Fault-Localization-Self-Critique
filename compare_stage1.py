"""
Compare what stage 1 (file localization) saw with vs without the issue text.

For each bug: was the real fix's file in the 25-file candidate shortlist, at what position,
and which file did the model pick first? Reads the stage-1 localization logs only.

Usage (agentless env):
    python compare_stage1.py                       # summary table, all 16 bugs
    python compare_stage1.py django__django-14672  # one bug in detail: both descriptions
                                                   # (first lines) + both shortlists
"""
import argparse
import json
import os
import re

from datasets import load_dataset

PILOT = ["django__django-12262", "django__django-13401", "django__django-14672",
         "django__django-15629", "django__django-15732", "pytest-dev__pytest-6202",
         "scikit-learn__scikit-learn-10908", "scikit-learn__scikit-learn-25973"]
ISSUE_DIRS = ["results/pilot_rerun", "results/multifile"]   # issue-text stage 1
NOISSUE_DIR = "results/swebench_noissue"


def gold_files(patch):
    return [b for _, b in re.findall(r"^diff --git a/(\S+) b/(\S+)", patch, re.M)]


def stage1(run_dir, bug):
    log = os.path.join(run_dir, bug, "localization_logs", f"{bug}.log")
    out = os.path.join(run_dir, bug, "loc_outputs.jsonl")
    if not os.path.isfile(out):
        return None
    found = json.loads(open(out, encoding="utf-8").readline()).get("found_files", [])
    text = open(log, encoding="utf-8", errors="replace").read() if os.path.isfile(log) else ""
    # some older logs hold several appended runs - use the LAST prompt; some (Sep 2026)
    # never logged the prompt at all -> desc/cands None ("not logged"), pick still known
    if "### Candidate Files ###" not in text:
        return None, None, found
    text = text[text.rindex("### GitHub Problem Description ###"):]
    desc = text.split("### GitHub Problem Description ###", 1)[1].split("###", 1)[0].strip()
    i = text.index("### Candidate Files ###") + len("### Candidate Files ###")
    cands = [l.strip() for l in text[i:text.index("###", i)].splitlines() if l.strip()]
    return desc, cands, found


def issue_stage1(bug):
    for d in ISSUE_DIRS:
        r = stage1(d, bug)
        if r:
            return r
    return None


def pos(lst, gold):
    hits = [lst.index(g) + 1 for g in gold if g in lst]
    return min(hits) if hits else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bug", nargs="?")
    args = ap.parse_args()

    ds = load_dataset("princeton-nlp/SWE-bench_Verified", split="test")
    bugs = sorted(os.listdir(NOISSUE_DIR))
    gold = {r["instance_id"]: gold_files(r["patch"]) for r in ds if r["instance_id"] in bugs}

    if args.bug:
        b = args.bug
        print(f"Real fix changed: {gold[b]}\n")
        for label, r in [("WITH ISSUE TEXT", issue_stage1(b)), ("WITHOUT (test report)", stage1(NOISSUE_DIR, b))]:
            desc, cands, found = r
            print(f"===== {label} =====")
            print("Description (first 15 lines):")
            print("\n".join("    " + l for l in desc.splitlines()[:15]))
            print("\nCandidate shortlist (* = real fix file):")
            for n, c in enumerate(cands, 1):
                print(f"  {'*' if c in gold[b] else ' '} {n:2}. {c}")
            print(f"\nModel picked: {found}\n")
        return

    print(f"{'bug':<34}{'group':<7}{'| gold in shortlist? (pos)':<27}{'| model 1st pick correct?'}")
    print(f"{'':<41}{'| issue   no-issue':<27}{'| issue   no-issue'}")
    tot = {"in_i": 0, "in_n": 0, "top_i": 0, "top_n": 0}
    for b in bugs:
        ri, rn = issue_stage1(b), stage1(NOISSUE_DIR, b)
        pi = pos(ri[1], gold[b]) if ri[1] else "n/l"
        pn = pos(rn[1], gold[b]) if rn[1] else "n/l"
        if pn == "n/l":
            pi = "n/l" if pi is None else pi
        ti = bool(ri[2]) and ri[2][0] in gold[b]
        tn = bool(rn[2]) and rn[2][0] in gold[b]
        if pi != "n/l" and pn != "n/l":
            tot["logged"] = tot.get("logged", 0) + 1
            tot["in_i"] += pi is not None; tot["in_n_l"] = tot.get("in_n_l", 0) + (pn is not None)
        if pn != "n/l":
            tot["n_logged"] = tot.get("n_logged", 0) + 1
            tot["in_n"] += pn is not None
        tot["top_i"] += ti; tot["top_n"] += tn
        f = lambda p: "not logged" if p == "n/l" else (f"yes ({p})" if p else "NO")
        print(f"{b:<34}{'pilot' if b in PILOT else 'multi':<7}| {f(pi):<11}{f(pn):<10}| "
              f"{'yes' if ti else '-':<8}{'yes' if tn else '-'}")
    n, L = len(bugs), tot.get("logged", 0)
    print(f"\nReal fix file reached the shortlist (bugs where both were logged, n={L}): "
          f"issue {tot['in_i']}/{L}, no-issue {tot.get('in_n_l', 0)}/{L}")
    print(f"  no-issue, all logged bugs: {tot['in_n']}/{tot.get('n_logged', 0)}")
    print(f"Model's first pick was a real fix file: issue {tot['top_i']}/{n}, no-issue {tot['top_n']}/{n}")


if __name__ == "__main__":
    main()
