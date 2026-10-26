"""
Convert BugsInPy bugs into the dataset format Agentless reads, so the SAME Agentless
pipeline (unmodified) can run on BugsInPy as on SWE-bench Verified.

Agentless needs four fields per bug: instance_id, repo, base_commit, problem_statement.
The first three come straight from BugsInPy's metadata. The fourth is the problem:
BugsInPy has NO issue text. Following AutoFL (Kang, An & Yoo, FSE 2024, Sec. 3 and
Listing 2), which ran on BugsInPy with the same gap, the problem statement is built from
the failing test instead:

  - the failing test's name and its code, cut at the failing statement (marked
    "# error occurred here"); the whole outermost statement is kept if the failure is
    inside a nested block; assertions BEFORE the failing statement are removed (they
    passed, so they are less relevant and can mislead) - in Python that means `assert`
    statements and assert*-named calls (self.assertEqual etc.), the counterpart of the
    JUnit assert* calls AutoFL removes; comment-only lines are not statements, so they
    do not appear;
  - the error message and call stack, keeping only frames inside the project (library
    and pytest frames dropped), with runs of an identical frame repeated more than five
    times condensed;
  - several failing tests are concatenated.
  AutoFL's final line ("Start by calling the get_failing_tests_covered_classes function")
  is NOT included: it tells the model to use AutoFL's tools, which Agentless does not have.

Getting the error and stack trace means RUNNING the triggering test on the buggy version,
so run this once per bug, in that bug's Python environment (e.g. conda env bugsinpy38):

    python bugsinpy_to_agentless.py ^
        --repo "C:/.../bugsinpy_work/black" ^
        --bug-dir "C:/.../BugsInPy/projects/black/bugs/6" ^
        --out datasets/bugsinpy_pilot

Each run adds (or replaces) one line in <out>/test.jsonl. Agentless then reads it with
`--dataset datasets/bugsinpy_pilot` (a local folder with test.jsonl loads as split "test").
Standard library only, so it runs in the old-Python BugsInPy environments.
"""
import argparse
import ast
import json
import os
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from run_bugsinpy_sbfl import (  # noqa: E402
    _is_test_path, prepare_checkout, read_bug_info, trigger_tests,
)


# ------------------------------------------------------------------ metadata
def read_project_url(bug_dir):
    """'owner/name' from the project's project.info github_url."""
    proj_info = os.path.join(bug_dir, "..", "..", "project.info")
    with open(proj_info, encoding="utf-8") as f:
        m = re.search(r'github_url="https?://github\.com/([^"]+?)/?"', f.read())
    if not m:
        sys.exit(f"no github_url in {proj_info}")
    return re.sub(r"\.git$", "", m.group(1))


def split_patch(bug_dir):
    """(source patch, test patch) from bug_patch.txt - BugsInPy sometimes mixes test
    helpers into the fix; Agentless's gold patch must be source-only, as in SWE-bench."""
    with open(os.path.join(bug_dir, "bug_patch.txt"), encoding="utf-8") as f:
        text = f.read()
    blocks = re.split(r"(?m)^(?=diff --git )", text)
    src, tst = [], []
    for b in blocks:
        m = re.match(r"diff --git a/(\S+)", b)
        if not m:
            continue
        (tst if _is_test_path(m.group(1)) else src).append(b)
    return "".join(src), "".join(tst)


# --------------------------------------------------------- running the tests
def to_node_id(repo, test_id):
    """BugsInPy test id -> pytest node id.
    'tests/test_x.py::C::t' stays; 'tests.test_x.C.t' (unittest style) is resolved by
    finding the longest dotted prefix that is a file in the repo."""
    if "::" in test_id:
        return test_id
    parts = test_id.split(".")
    for i in range(len(parts) - 1, 0, -1):
        path = "/".join(parts[:i]) + ".py"
        if os.path.exists(os.path.join(repo, path)):
            return path + "::" + "::".join(parts[i:])
    sys.exit(f"cannot map test id to a file: {test_id}")


def _match_node(nodes, junit_class, junit_name):
    """The node id a JUnit (classname, name) pair came from.
    'tests/test_x.py::C::t' is reported as classname 'tests.test_x.C', name 't'."""
    name = re.sub(r"\[.*\]$", "", junit_name)
    for n in nodes:
        path, *names = n.split("::")
        module = path[:-3].replace("/", ".")
        expected = ".".join([module] + names[:-1])
        if names and names[-1] == name and junit_class == expected:
            return n
    return None


def _end(node, siblings, src_lines=None):
    """Last line of an AST statement. Python 3.8+ has end_lineno. Python 3.7 (some
    BugsInPy bugs) does not, so use the last line any part of the statement is on, then
    take in lines that only close brackets (e.g. a call's final ')'). Never past the
    next sibling statement or the end of the file."""
    if getattr(node, "end_lineno", None):
        return node.end_lineno
    end = max(getattr(n, "lineno", 0) for n in ast.walk(node))
    if src_lines:
        later = [s.lineno for s in siblings if s.lineno > node.lineno]
        limit = min(later) - 1 if later else len(src_lines)
        while end < limit and src_lines[end].strip()[:1] in (")", "]", "}") \
                and src_lines[end].strip():
            end += 1
        end = min(end, len(src_lines))
    return end


def run_triggers(repo, nodes):
    """Run only the triggering tests; return {node: (message, native traceback)}."""
    junit = os.path.join(tempfile.mkdtemp(), "junit.xml")
    subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=native",
         f"--junitxml={junit}", *nodes],
        cwd=repo, check=False, capture_output=True, text=True,
    )
    if not os.path.exists(junit):
        sys.exit("pytest produced no junit.xml - the test run did not start")
    out = {}
    for case in ET.parse(junit).iter("testcase"):
        bad = case.find("failure")
        if bad is None:
            bad = case.find("error")
        if bad is None:
            continue
        cls, name = case.get("classname") or "", case.get("name") or ""
        if not cls or "collection failure" in (bad.get("message") or ""):
            detail = next((l.strip() for l in (bad.text or "").splitlines()
                           if "Error" in l), "")
            hint = ("\nThe test imports something the BUGGY code does not have yet - the fix\n"
                    "probably ADDS it (feature/refactor, not a bug fix). Drop this bug."
                    if "cannot import name" in (bad.text or "") else
                    "\nEnvironment problem, not a result. Fix the environment and rerun.")
            sys.exit(f"SETUP ERROR - test module could not be loaded: {name}\n  {detail}{hint}")
        # pytest's XML leaves some characters as '#x1F370' escapes - restore them
        tb = re.sub(r"&?#x([0-9A-Fa-f]{2,6});?", lambda m: chr(int(m.group(1), 16)),
                    bad.text or "")
        out[_match_node(nodes, cls, name) or f"{cls}.{name}"] = (bad.get("message") or "", tb)
    return out


# -------------------------------------------------- AutoFL-style minimisation
FRAME_RE = re.compile(r'^\s*File "([^"]+)", line (\d+), in (\S+)')


def minimise_trace(native_tb, repo):
    """Keep only frames inside the project, plus the final exception line(s)."""
    lines = native_tb.splitlines()
    repo_norm = os.path.normcase(os.path.abspath(repo))
    frames, exc, i = [], [], 0
    while i < len(lines):
        m = FRAME_RE.match(lines[i])
        if m:
            path = os.path.normcase(os.path.abspath(m.group(1)))
            code = lines[i + 1].strip() if i + 1 < len(lines) and not FRAME_RE.match(lines[i + 1]) else ""
            if path.startswith(repo_norm) and "site-packages" not in path:
                rel = os.path.relpath(m.group(1), repo).replace("\\", "/")
                frames.append((rel, int(m.group(2)), m.group(3), code))
            i += 2 if code else 1
            continue
        if frames:
            exc.append(lines[i].rstrip())          # everything after the last frame
        i += 1
    # the error message = from the exception line (e.g. "AssertionError: ...") to the end,
    # INCLUDING multi-line messages such as assertEqual diffs; long ones are capped
    start = next((k for k, l in enumerate(exc)
                  if re.match(r"^[A-Za-z_][\w.]*(Error|Exception|Exit|Interrupt|Warning|\w*)(:|$)", l)
                  and not l.startswith((" ", "\t"))), None)
    exc = [l for l in (exc[start:] if start is not None else exc) if l.strip() or l == ""]
    while exc and not exc[-1].strip():
        exc.pop()
    if len(exc) > 25:
        exc = exc[:25] + [f"... ({len(exc) - 25} more lines of the error message)"]
    # condense runs of an identical frame repeated more than five times
    condensed, run = [], 1
    for j, fr in enumerate(frames):
        if j and fr == frames[j - 1]:
            run += 1
            if run > 5:
                continue
        else:
            if run > 5:
                condensed.append(("...", 0, f"(previous frame repeated {run - 5} more times)", ""))
            run = 1
        condensed.append(fr)
    if run > 5:
        condensed.append(("...", 0, f"(previous frame repeated {run - 5} more times)", ""))
    body = []
    for rel, ln, fn, code in condensed:
        if rel == "...":
            body.append(f"  {fn}")
            continue
        body.append(f'  File "{rel}", line {ln}, in {fn}')
        if code:
            body.append(f"    {code}")
    exc_text = "\n".join(exc)
    return exc_text, "Traceback (most recent call last):\n" + "\n".join(body) + (
        "\n" + exc_text if exc_text else "")


def _is_assertion(stmt):
    """An assertion statement: Python's `assert`, or a call whose name starts with
    'assert' (unittest's self.assertEqual, helpers like assert_stable) - the Python
    counterpart of the JUnit assert* calls AutoFL removes."""
    if isinstance(stmt, ast.Assert):
        return True
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        f = stmt.value.func
        name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", "")
        return name.lower().startswith("assert")
    return False


def test_snippet(repo, node, trace_frames_text):
    """The test function, numbered, cut at the failing statement (AutoFL heuristics)."""
    path, *names = node.split("::")
    src_path = os.path.join(repo, path)
    with open(src_path, encoding="utf-8", errors="replace") as f:
        source = f.read()
    src_lines = source.splitlines()
    tree = ast.parse(source)
    fn, scope = None, tree.body
    for nm in names:
        nm = re.sub(r"\[.*\]$", "", nm)
        parent_scope = scope
        fn = next((n for n in scope if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                                                      ast.ClassDef)) and n.name == nm), None)
        if fn is None:
            return None
        scope = fn.body
    fn_end = _end(fn, parent_scope, src_lines)
    # failing line in THIS test file = the last frame of the trace that is in it
    fail_line = None
    for m in re.finditer(r'File "([^"]+)", line (\d+)', trace_frames_text):
        if m.group(1).replace("\\", "/").endswith(path):
            ln = int(m.group(2))
            if fn.lineno <= ln <= fn_end:
                fail_line = ln
    start = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno)
    keep = list(range(start, fn.body[0].lineno))          # decorators + def line
    for stmt in fn.body:
        stmt_end = min(_end(stmt, fn.body, src_lines), fn_end)
        # a decorated def/class starts at its first decorator, not its `def` line
        stmt_start = min([stmt.lineno] + [d.lineno for d in getattr(stmt, "decorator_list", [])])
        hit = fail_line is not None and stmt_start <= fail_line <= stmt_end
        if not hit and _is_assertion(stmt):
            continue                                       # passed assertion: drop
        keep.extend(range(stmt_start, stmt_end + 1))
        if hit:
            break
    out = []
    for ln in keep:
        text = src_lines[ln - 1]
        if ln == fail_line:
            text += "  # error occurred here"
        out.append(f"{ln} : {text}")
    return "\n".join(out)


def build_problem_statement(repo, results):
    parts = []
    for node, (message, tb) in results.items():
        exc_text, trace = minimise_trace(tb, repo)
        snippet = test_snippet(repo, node, tb) if "::" in node else None
        block = f"The test `{node}` failed."
        if snippet:
            block += f" The test looks like:\n\n```python\n{snippet}\n```\n"
        block += f"\nIt failed with the following error message and call stack:\n\n```\n{trace}\n```"
        parts.append(block)
    return "\n\n".join(parts)


# ----------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="local clone of the BugsInPy project")
    ap.add_argument("--bug-dir", help="BugsInPy projects/<p>/bugs/<n> folder")
    ap.add_argument("--out", default="datasets/bugsinpy_pilot")
    ap.add_argument("--drop", metavar="INSTANCE_ID",
                    help="remove one bug (e.g. bugsinpy__PySnooper-2) from <out>/test.jsonl")
    args = ap.parse_args()

    if args.drop:
        path = os.path.join(args.out, "test.jsonl")
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
        kept = [r for r in rows if r["instance_id"] != args.drop]
        with open(path, "w", encoding="utf-8") as f:
            for r in kept:
                f.write(json.dumps(r) + "\n")
        print(f"Removed {len(rows) - len(kept)} row(s) for {args.drop}; {len(kept)} bug(s) left in {path}")
        return
    if not (args.repo and args.bug_dir):
        ap.error("--repo and --bug-dir are required (unless using --drop)")

    repo = os.path.abspath(args.repo)
    info = read_bug_info(args.bug_dir)
    project = os.path.basename(os.path.dirname(os.path.dirname(os.path.abspath(args.bug_dir))))
    bug_no = os.path.basename(os.path.abspath(args.bug_dir))
    instance_id = f"bugsinpy__{project}-{bug_no}"

    prepare_checkout(repo, info)                 # buggy source + fix's test files
    nodes = [to_node_id(repo, t) for t in trigger_tests(args.bug_dir)]
    if not nodes:
        sys.exit("no triggering test in run_test.sh")
    print(f"Running triggering test(s): {nodes}")
    results = run_triggers(repo, nodes)
    if not results:
        sys.exit("The triggering test(s) PASSED on the buggy version - nothing to report. "
                 "Check the checkout/overlay before using this bug.")
    missing = [n for n in nodes if n not in results]
    if missing:
        print(f"WARNING: these triggering tests did not fail: {missing}")

    # CRASH/LOGIC label from the actual failure (rule in capstone/bugsinpy_summary.md):
    # AssertionError = LOGIC (wrong value), any other exception = CRASH. With several
    # failing tests, any crash makes the bug CRASH.
    exc_types = {}
    for node, (message, tb) in results.items():
        last = next((l for l in reversed((tb or message).splitlines())
                     if re.match(r"^[\w.]+(Error|Exception|Exit|Interrupt|\w+)(:|$)", l)), message)
        exc_types[node] = last.split(":")[0].strip().split(".")[-1] or "unknown"
    category = "LOGIC" if all(t == "AssertionError" for t in exc_types.values()) else "CRASH"
    print(f"Failure types: {exc_types} -> {category}")
    # A test that calls something the buggy code does not HAVE yet is a feature addition,
    # not a bug (SWE-bench labels: FEATURE - excluded from the experiment, no fault to
    # localize). The error type alone would wrongly call it CRASH. Flag it for a manual look.
    feature_signs = (r"unexpected keyword argument", r"has no attribute",
                     r"cannot import name", r"No module named", r"takes \d+ positional")
    hits = [n for n, (m, tb) in results.items()
            if any(re.search(s, (m or "") + "\n" + (tb or "")) for s in feature_signs)]
    if hits:
        print("\n*** WARNING: possible FEATURE ADDITION, not a bug ***\n"
              f"    {hits} fail because the buggy code is missing something the test calls.\n"
              "    Read the fix (bug_patch.txt) before using this bug; drop it if the fix adds\n"
              "    new behaviour rather than correcting existing behaviour.\n")

    src_patch, test_patch = split_patch(args.bug_dir)
    record = {
        "instance_id": instance_id,
        "repo": read_project_url(args.bug_dir),
        "base_commit": info["buggy_commit_id"],
        "problem_statement": build_problem_statement(repo, results),
        "patch": src_patch,                      # gold, source files only
        "test_patch": test_patch,
        "FAIL_TO_PASS": json.dumps(nodes),
        "PASS_TO_PASS": "[]",
        "hints_text": "",
        "created_at": "",
        "version": "",
        "environment_setup_commit": info["buggy_commit_id"],
        "category": category,
        "failure_types": json.dumps(exc_types),
        "bugsinpy_project": project,
        "bugsinpy_bug": bug_no,
        "python_version": info.get("python_version", ""),
        "fixed_commit": info["fixed_commit_id"],
        "problem_statement_source": "AutoFL-style failing-test report (Kang et al., FSE 2024)",
    }

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "test.jsonl")
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            rows = [json.loads(l) for l in f if l.strip()]
    rows = [r for r in rows if r["instance_id"] != instance_id] + [record]
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"\nWrote {instance_id} -> {path}  ({len(rows)} bug(s) in the file)")
    print("\n----- problem_statement given to Agentless -----")
    # Windows consoles can't print some characters (black's output has emoji)
    enc = sys.stdout.encoding or "utf-8"
    print(record["problem_statement"].encode(enc, errors="replace").decode(enc))


if __name__ == "__main__":
    main()
