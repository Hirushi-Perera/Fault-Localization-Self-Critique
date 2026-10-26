"""
Ablation: remove ONE clue from the no-issue failing-test report, re-run stage 1 (file-level
localization), and see whether the model's top file changes. At temperature 0 the same input
gives the same answer (checked by the control run), so a change means that clue caused the pick.

Variants (all built from datasets/swebench_noissue, nothing else changed):
  no_fixframes   CRASH bugs found WITH the fix file in the trace: delete every trace frame
                 that points into a fix file (the "File ..." line and the code line under it).
                 Tests: was the trace really what led the model to the file?
  no_testpaths   bugs found WITHOUT any fix file in the trace: replace every test file path
                 with "<test file>". Tests: did the test file NAME give the location away?
  control        two bugs, input unchanged. Tests: does a re-run reproduce the same picks?

Usage (agentless env, from the Agentless folder):
    python ablation_stage1.py make       # writes datasets/ablation_<variant>/test.jsonl, shows what changed
    (run stage 1 - see run_ablation.ps1)
    python ablation_stage1.py compare    # original vs ablated top picks
"""
import json
import os
import re
import sys

SRC = "datasets/swebench_noissue/test.jsonl"
ORIG_RUN = "results/swebench_noissue"
VARIANTS = {
    "no_fixframes": ["django__django-10554", "django__django-11532", "django__django-11734",
                     "django__django-12262", "django__django-13121",
                     "scikit-learn__scikit-learn-25973"],
    "no_testpaths": ["scikit-learn__scikit-learn-10908", "django__django-11400",
                     "django__django-13195"],
    "control": ["django__django-11734", "django__django-11138"],
}


def gold_files(patch):
    return re.findall(r"^diff --git a/\S+ b/(\S+)", patch, re.M)


def is_test(path):
    return "/tests/" in path or "/test_" in path or path.startswith(("tests/", "testing/"))


def drop_fix_frames(text, gold):
    out, skip_code = [], False
    for line in text.splitlines():
        m = re.match(r'\s*File "([^"]+)"', line)
        if m:
            skip_code = any(m.group(1) == g or m.group(1).endswith("/" + g) for g in gold)
            if skip_code:
                continue
        elif skip_code and line.startswith("    "):        # the code line under that frame
            skip_code = False
            continue
        else:
            skip_code = False
        out.append(line)
    return "\n".join(out)


def hide_test_paths(text):
    paths = set(re.findall(r'File "([^"]+)"', text)) | set(re.findall(r"`([^`:]+\.py)::", text))
    for p in sorted((p for p in paths if is_test(p)), key=len, reverse=True):
        text = text.replace(p, "<test file>")
    return text


def make():
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in open(SRC, encoding="utf-8")}
    for variant, bugs in VARIANTS.items():
        out_dir = f"datasets/ablation_{variant}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "test.jsonl"), "w", encoding="utf-8") as f:
            for b in bugs:
                r = dict(rows[b])
                before = r["problem_statement"]
                if variant == "no_fixframes":
                    r["problem_statement"] = drop_fix_frames(before, gold_files(r["patch"]))
                elif variant == "no_testpaths":
                    r["problem_statement"] = hide_test_paths(before)
                f.write(json.dumps(r) + "\n")
                removed = [l for l in before.splitlines() if l not in r["problem_statement"].splitlines()]
                print(f"[{variant}] {b}: {len(removed)} lines changed/removed")
                for l in removed[:3]:
                    print(f"      - {l.strip()[:110]}")
    print("\nDatasets written. Next: .\\run_ablation.ps1")


def top(run, bug):
    p = os.path.join(run, bug, "loc_outputs.jsonl")
    if not os.path.isfile(p):
        return None
    return (json.loads(open(p, encoding="utf-8").readline()).get("found_files") or ["-"])


def compare():
    rows = {json.loads(l)["instance_id"]: json.loads(l) for l in open(SRC, encoding="utf-8")}
    for variant, bugs in VARIANTS.items():
        print(f"\n=== {variant}")
        for b in bugs:
            gold = gold_files(rows[b]["patch"])
            o, a = top(ORIG_RUN, b), top(f"results/ablation_{variant}", b)
            if a is None:
                print(f"  {b:<34} not run yet")
                continue
            fo, fa = o[0] in gold, a[0] in gold
            verdict = ("same pick" if o[0] == a[0] else
                       "LOST the file" if fo and not fa else
                       "gained the file" if fa and not fo else "different pick")
            print(f"  {b:<34} {verdict}")
            print(f"      original : {o[0]}{' *' if fo else ''}")
            print(f"      ablated  : {a[0]}{' *' if fa else ''}   (top 3: {a[:3]})")
    print("\n  * = a file the real fix changed")


if __name__ == "__main__":
    {"make": make, "compare": compare}[sys.argv[1] if len(sys.argv) > 1 else "make"]()
