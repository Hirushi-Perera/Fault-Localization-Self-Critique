"""
Build a SWE-bench dataset folder in which each bug's GitHub issue text is REPLACED by the
same AutoFL-style failing-test report that BugsInPy bugs get (bugsinpy_to_agentless.py):
the failing test's code cut at the failing statement, plus the error message and the call
stack restricted to project frames. Everything else in the SWE-bench row (repo,
base_commit, gold patch, tests) is kept, so the same Agentless chain and scorers run
unchanged. Purpose (supervisor, 12 Oct 2026): measure how much the issue text itself helps
localization, and make SWE-bench comparable with BugsInPy, which has no issue text.

Getting the error and call stack means RUNNING each bug's FAIL_TO_PASS tests on the
BUGGY code. That happens in the bug's official SWE-bench Docker image, so first build the
images by running the harness once with a no-op patch (see make_noop_preds.py); the
instance images it leaves behind (sweb.eval.x86_64.*) are reused here. The harness's own
eval.sh is replayed with two changes: pytest gets --tb=native (Python-style call stacks,
as BugsInPy's reports use), and the test files are printed (base64) after the run so the test code
can be cut at the failing line.

Run inside WSL (needs the docker CLI and the `datasets` package, e.g. ~/swe/bin/python):

    ~/swe/bin/python swebench_to_failing_test.py \
        --logs ~/swebench_runs/logs/run_evaluation \
        --out datasets/swebench_noissue \
        django__django-12262 pytest-dev__pytest-6202 ...
"""
import argparse
import base64
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from bugsinpy_to_agentless import minimise_trace, test_snippet  # noqa: E402

REPO_ROOT = "/testbed"                 # where SWE-bench images keep the repository
FILE_MARK = "@@@SWEBENCH_TEST_FILE "


# ------------------------------------------------------------- running the tests
def image_name(instance_id):
    return "swebench/sweb.eval.x86_64." + instance_id.replace("__", "_1776_") + ":latest"


def find_eval_script(logs, instance_id):
    hits = glob.glob(os.path.join(os.path.expanduser(logs), "*", "*", instance_id, "eval.sh"))
    if not hits:
        sys.exit(f"no harness eval.sh for {instance_id} under {logs} - run the harness first")
    return hits[0]


def patch_files(test_patch):
    return re.findall(r"(?m)^diff --git a/(\S+) b/", test_patch)


def replay_script(eval_sh, test_files):
    """The harness eval.sh, with native pytest tracebacks and the test files dumped
    after the tests ran (before eval.sh restores them)."""
    with open(eval_sh, encoding="utf-8") as f:
        lines = f.read().splitlines()
    out, after_tests = [], False
    for line in lines:
        if after_tests and line.startswith("git checkout "):
            out.append("set +x")                   # keep the dump free of xtrace lines
            for p in test_files:
                out.append(f"echo \"{FILE_MARK}{p} $(base64 -w0 '{p}' 2>/dev/null)\"")
            after_tests = False
        if re.match(r"^(python -m )?pytest\b", line) and "--tb=" not in line:
            line = re.sub(r"^((?:python -m )?pytest)\b", r"\1 --tb=native", line)
        if line.startswith("echo \">>>>> Test Exit Code"):
            after_tests = True
        out.append(line)
    return "\n".join(out) + "\n"


def run_in_image(instance_id, script):
    proc = subprocess.run(["docker", "run", "--rm", "-i", image_name(instance_id), "bash", "-s"],
                          input=script, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors="replace",
                          timeout=3600)
    if "Start Test Output" not in proc.stdout:
        sys.exit(f"{instance_id}: tests did not run\n{proc.stdout[-2000:]}")
    return proc.stdout


def split_output(stdout):
    """(test output, {test file path: source after the test patch})"""
    test_out = stdout.split(">>>>> Start Test Output", 1)[1].split(">>>>> End Test Output", 1)[0]
    test_out = re.sub(r"\x1b\[[0-9;]*m", "", test_out)          # pytest colour codes
    test_out = "\n".join(l for l in test_out.splitlines()        # drop bash xtrace lines
                         if not l.startswith("+ ") and l.strip() not in ("'", "+ : '"))
    files = {m.group(1): base64.b64decode(m.group(2)).decode("utf-8", "replace")
             for m in re.finditer(r"(?m)^" + FILE_MARK + r"(\S+) (\S*)$", stdout)}
    return test_out, files


# --------------------------------------------------- failing tests -> tracebacks
DJANGO_HEAD = re.compile(r"^(ERROR|FAIL): (\w+) \(([\w.]+)\)", re.M)
PYTEST_HEAD = re.compile(r"^_{3,} (.+?) _{3,}$", re.M)


def django_failures(test_out):
    """{test id: traceback} from Django's runner output. A test is keyed both as
    'test_name (module.Class)' and, when it has a docstring, by the docstring's first
    line - SWE-bench uses the docstring as the FAIL_TO_PASS id for such tests."""
    res = {}
    for b in re.split(r"(?m)^={50,}\s*$", test_out):
        m = DJANGO_HEAD.search(b)
        if not m:
            continue
        name, where = m.group(2), m.group(3)
        if where.endswith("." + name):            # newer unittest: (module.Class.test)
            where = where[: -len(name) - 1]
        rest = "\n" + b[m.end():].partition("\n")[2]
        doc, _, tb = rest.partition("\n-")        # optional docstring line, then dashes
        tb = tb.partition("\n")[2]                 # drop the rest of the dashes line
        tb = re.split(r"(?m)^-{50,}\s*\nRan \d+ test", tb)[0].strip("\n")
        val = (tb, name, where.split(".")[-1])
        res.setdefault(f"{name} ({where})", val)
        if doc.strip():
            res.setdefault(doc.strip(), val)
    return res


def pytest_failures(test_out):
    """{'Class.test_name[param]' or 'test_name[param]': traceback} from pytest -rA."""
    res = {}
    heads = list(PYTEST_HEAD.finditer(test_out))
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else len(test_out)
        body = test_out[m.end():end]
        body = re.split(r"(?m)^(-{3,} Captured|={3,} )", body)[0]
        if "Traceback (most recent call last)" in body:
            res.setdefault(m.group(1).strip(), (body.strip("\n"), None, None))
    return res


def match_failures(f2p, failures, kind):
    """{F2P test id: (traceback, test name, class)} for the FAIL_TO_PASS tests that failed."""
    out = {}
    for t in f2p:
        if kind == "django":
            if t in failures:
                out[t] = failures[t]
            continue
        path, *names = t.split("::")
        dotted = ".".join(names)
        for title, val in failures.items():
            if title == dotted or title == names[-1] or dotted.endswith("." + title):
                out[t] = val
                break
    return out


def runner_crash(f2p, test_out):
    """Django's runner can itself crash, so no ERROR/FAIL block is printed:
    - while printing a failure (django-10554: a subTest label's repr() re-runs the buggy
      query) - the call stack is used for the FAIL_TO_PASS test that did not end in "ok";
    - before any test runs (django-14672: the model checks hit the bug at start-up, so all
      168 FAIL_TO_PASS tests fail together) - the call stack is reported once, for the
      test module, like a pytest collection error."""
    pos = test_out.rfind("Traceback (most recent call last)")
    if pos < 0:
        return {}
    tb = test_out[pos:].strip("\n")
    if not any(t in test_out[:pos] for t in f2p):
        module = re.match(r"\w+ \(([\w.]+)\)", f2p[0]).group(1).rsplit(".", 1)[0]
        return {"tests/" + module.replace(".", "/") + ".py": (tb, None, None)}
    # verbose lines read "test_x (module.Class) ... ok"; the one with no "ok" crashed
    not_ok = [t for t in f2p if t in test_out[:pos]
              and not test_out[test_out.find(t) + len(t):].startswith(" ... ok")]
    if len(not_ok) != 1:
        return {}
    running = not_ok[0]
    name, where = re.match(r"(\w+) \(([\w.]+)\)", running).groups()
    return {running: (test_out[pos:].strip("\n"), name, where.split(".")[-1])}


def node_for(test_id, name, cls, tb, kind, files):
    """A pytest-style node id (path::Class::test) for the report, like BugsInPy's: the
    test file is the call-stack frame running the test, else the patched test file that
    defines both the class and the method."""
    if kind != "django" or name is None:          # pytest id, or a whole test module
        return test_id
    for m in re.finditer(r'File "([^"]+)", line \d+, in (\w+)', tb):
        if m.group(2) == name and m.group(1).startswith(REPO_ROOT + "/"):
            return f"{os.path.relpath(m.group(1), REPO_ROOT)}::{cls}::{name}"
    for path, src in files.items():
        if re.search(rf"(?m)^class {cls}\b", src) and re.search(rf"(?m)^\s+def {name}\(", src):
            return f"{path}::{cls}::{name}"
    return None


# ----------------------------------------------------------------------- report
def build_report(results, files):
    """Same wording and layout as bugsinpy_to_agentless.build_problem_statement."""
    with tempfile.TemporaryDirectory() as tmp:
        for path, body in files.items():               # the test files, for the snippet
            dest = os.path.join(tmp, path)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(body)
        parts = []
        for node, tb in results:
            _, trace = minimise_trace(tb, REPO_ROOT)
            snippet = None
            if "::" in node and node.split("::")[0] in files:
                try:
                    snippet = test_snippet(tmp, node, tb.replace(REPO_ROOT + "/", ""))
                except SyntaxError:
                    snippet = None
            block = f"The test `{node}` failed."
            if snippet:
                block += f" The test looks like:\n\n```python\n{snippet}\n```\n"
            block += f"\nIt failed with the following error message and call stack:\n\n```\n{trace}\n```"
            parts.append(block)
        return "\n\n".join(parts)


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("instance_ids", nargs="+")
    ap.add_argument("--logs", required=True, help="harness logs/run_evaluation folder")
    ap.add_argument("--out", default="datasets/swebench_noissue")
    ap.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified",
                    help="the copy Agentless itself loads (test ids stored as JSON strings)")
    args = ap.parse_args()

    from datasets import load_dataset
    rows = {r["instance_id"]: r for r in load_dataset(args.dataset, split="test")
            if r["instance_id"] in set(args.instance_ids)}

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "test.jsonl")
    kept = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            kept = [json.loads(l) for l in f if l.strip()]

    for iid in args.instance_ids:
        row = dict(rows[iid])
        kind = "django" if row["repo"] == "django/django" else "pytest"
        f2p = row["FAIL_TO_PASS"]
        f2p = json.loads(f2p) if isinstance(f2p, str) else list(f2p)
        script = replay_script(find_eval_script(args.logs, iid), patch_files(row["test_patch"]))
        test_out, files = split_output(run_in_image(iid, script))
        failures = django_failures(test_out) if kind == "django" else pytest_failures(test_out)
        matched = match_failures(f2p, failures, kind)
        if not matched and kind == "django":
            matched = runner_crash(f2p, test_out)
            if matched:
                print(f"   {iid}: the test runner crashed (no per-test failure block); "
                      f"its call stack is reported for {list(matched)[0]}")
        if not matched:
            print(f"!! {iid}: none of the FAIL_TO_PASS tests failed with a traceback - skipped")
            continue
        missing = [t for t in f2p if t not in matched]
        if all(k.endswith(".py") for k in matched):      # start-up crash: whole module
            missing = []
        results = []
        for t, (tb, name, cls) in matched.items():
            node = node_for(t, name, cls, tb, kind, files) or t
            results.append((node, tb))
        row["problem_statement"] = build_report(results, files)
        row["hints_text"] = ""
        row["problem_statement_source"] = ("AutoFL-style failing-test report (Kang et al., "
                                           "FSE 2024), replacing the SWE-bench issue text")
        row["report_tests"] = json.dumps([n for n, _ in results])
        kept = [r for r in kept if r["instance_id"] != iid] + [row]
        print(f"OK {iid}: {len(matched)}/{len(f2p)} FAIL_TO_PASS tests in the report"
              + (f" (no traceback for {len(missing)})" if missing else ""))

    with open(path, "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r) + "\n")
    print(f"\n{len(kept)} bug(s) in {path}")


if __name__ == "__main__":
    main()
