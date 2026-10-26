"""
Print the error message + call stack the model saw in the no-issue run, with the real fix
file(s) and which trace lines point into source code (not tests).

Usage (agentless env, from the Agentless folder):
    python show_traceback.py                         # the 4 bugs in the CRASH/LOGIC pairs
    python show_traceback.py django__django-11734    # any bugs
"""
import json
import re
import sys

PAIRS = ["django__django-11138", "django__django-11734",
         "django__django-13195", "django__django-10554"]


def main():
    bugs = sys.argv[1:] or PAIRS
    rows = {}
    with open("datasets/swebench_noissue/test.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["instance_id"] in bugs:
                rows[r["instance_id"]] = r
    for b in bugs:
        r = rows[b]
        gold = re.findall(r"^diff --git a/\S+ b/(\S+)", r["patch"], re.M)
        traces = re.findall(r"call stack:\n\n```\n(.*?)\n```", r["problem_statement"], re.S)
        print(f"\n{'=' * 80}\n{b}\nReal fix file(s): {gold}")
        for n, t in enumerate(traces, 1):
            print(f"\n--- trace {n} of {len(traces)} ---")
            for l in t.splitlines():
                m = re.search(r'File "([^"]+)"', l)
                tag = ""
                if m:
                    p = m.group(1)
                    tag = ("   <== FIX FILE" if any(p.endswith(g) for g in gold) else
                           "   (test code)" if "/tests/" in p or "/test_" in p or p.startswith("tests/")
                           else "   (source)")
                print(l + tag)
        src = sum(1 for t in traces for p in re.findall(r'File "([^"]+)"', t)
                  if "/tests/" not in p and "/test_" not in p and not p.startswith("tests/"))
        print(f"\nSource (non-test) frames: {src}; fix file in trace: "
              f"{any(g in t for g in gold for t in traces)}")


if __name__ == "__main__":
    main()
