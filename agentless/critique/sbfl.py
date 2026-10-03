"""
Spectrum-based fault localization (SBFL) with the Ochiai coefficient.

Turns test execution into a ranked list of suspicious FUNCTIONS. This is the execution
evidence the grounded critique condition is given. No LLM anywhere in this file.

Ported from the project sketch (capstone/sbfl_sketch.py), verified end to end in Phase 1
on a toy repo and on a real BugsInPy bug (PySnooper bug 3).

PIPELINE
  1. Set up the buggy repo state          (buggy source + the triggering tests, NO fix)
  2. Run tests with per-test coverage     (which test touched which line)
  3. Build the spectrum                   (per line: ef, ep)
  4. Score each line with Ochiai
  5. Map lines -> functions via `ast`
  6. Aggregate to function level (max), rank, keep top N

CRITICAL SETUP POINT
  SBFL must run on the BUGGY code - we are trying to locate the bug. The test that exposes
  the bug usually does not exist at the buggy commit (it is added with the fix), so:
      SWE-bench : checkout base_commit, apply test_patch, do NOT apply patch
      BugsInPy  : checkout buggy_commit, overlay the test file from fixed_commit
  Then the failing tests fail and the passing tests pass - the split Ochiai needs.
"""

import ast
import math
import os
import re
from collections import defaultdict

TEST_DIR_NAMES = {"test", "tests", "testing"}


# ============================================================ 1. THE FORMULA
def ochiai(ef, ep, total_failing):
    """
    Abreu, Zoeteweij & van Gemund, "On the Accuracy of Spectrum-based Fault
    Localization", TAICPART-MUTATION 2007.

        ochiai = ef / sqrt(total_failing * (ef + ep))

      ef            = number of FAILING tests that execute this line
      ep            = number of PASSING tests that execute this line
      total_failing = total number of failing tests   (== ef + nf)

    A line never touched by a failing test cannot be the fault -> score 0.
    """
    if ef == 0:
        return 0.0
    denom = math.sqrt(total_failing * (ef + ep))
    return ef / denom if denom else 0.0


# ================================================ 2. LINE -> FUNCTION MAPPING
def build_line_to_function_map(source_path):
    """
    Parse a Python file and return {line_number: "Class.method" or "function"}.
    Inner definitions are applied after outer ones, so the most specific (nested)
    name wins for a given line - e.g. a closure defined inside a function.
    """
    try:
        with open(source_path, encoding="utf-8", errors="ignore") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, OSError):
        return {}

    spans = []   # (depth, start, end, qualified_name)

    def walk(node, class_stack, depth):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                walk(child, class_stack + [child.name], depth + 1)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualified = ".".join(class_stack + [child.name])
                end = getattr(child, "end_lineno", child.lineno)
                spans.append((depth, child.lineno, end, qualified))
                walk(child, class_stack, depth + 1)
            else:
                walk(child, class_stack, depth)

    walk(tree, [], 0)

    line_map = {}
    for depth, start, end, name in sorted(spans, key=lambda s: s[0]):
        for ln in range(start, end + 1):
            line_map[ln] = name
    return line_map


# ======================================================= 3. TEST-NAME MATCHING
def _segments(test_id):
    """
    Normalise a test identifier to its dotted name segments, so identifiers from
    different sources compare equal:

        coverage context   "tests.test_x.TestC.test_m|run"
        pytest node id     "tests/test_x.py::TestC::test_m[case1]"
        JUnit class+name   "tests.test_x.TestC" + "test_m"
        bare name          "test_m"

    Parametrisation ([...]) is dropped: coverage's test_function context records the
    function, not the individual parameter case.
    """
    s = test_id.split("|")[0].strip()
    s = re.sub(r"\[.*\]$", "", s)                 # drop pytest parametrisation
    s = s.replace("\\", "/").replace("::", ".")
    s = re.sub(r"\.py(?=\.|$)", "", s)              # tests/test_x.py -> tests/test_x
    s = s.replace("/", ".")
    return [p for p in s.split(".") if p]


def same_test(a, b):
    """
    True when two test identifiers name the same test: one's segments must be a SUFFIX
    of the other's, at segment boundaries.

    Replaces the sketch's substring check (`a in b or b in a`), under which the
    failing test `test_add` also matched a PASSING test `test_add_user` and
    corrupted the spectrum. Segment matching compares whole names only.
    """
    sa, sb = _segments(a), _segments(b)
    if not sa or not sb:
        return False
    n = min(len(sa), len(sb))
    return sa[-n:] == sb[-n:]


def _is_project_source(rel):
    """Source file we should score: .py, inside the project, not a test file."""
    rel = rel.replace("\\", "/")
    if not rel.endswith(".py") or rel.startswith("..") or "site-packages" in rel:
        return False
    parts = rel.split("/")
    if any(p in TEST_DIR_NAMES for p in parts[:-1]):
        return False
    name = parts[-1]
    return not (name.startswith("test_") or name.endswith("_test.py")
                or name == "conftest.py")


# ================================================== 4. COVERAGE -> SPECTRUM
def build_spectrum(coverage_data_file, failing_tests, passing_tests, project_root):
    """
    Read a coverage.py data file recorded with per-test contexts and return
    {(file, line): {"ef": int, "ep": int}}, file paths relative to project_root with
    forward slashes (so they compare directly with gold-patch paths).

    Per-test coverage needs dynamic contexts, in .coveragerc:

        [run]
        dynamic_context = test_function
    """
    from coverage import CoverageData

    data = CoverageData(basename=coverage_data_file)
    data.read()

    def classify(context):
        """Map a coverage context string to 'fail', 'pass', or None."""
        if not context.split("|")[0]:
            return None
        if any(same_test(context, t) for t in failing_tests):
            return "fail"
        if any(same_test(context, t) for t in passing_tests):
            return "pass"
        return None

    spectrum = defaultdict(lambda: {"ef": 0, "ep": 0})

    for filename in data.measured_files():
        rel = os.path.relpath(filename, project_root).replace("\\", "/")
        if not _is_project_source(rel):
            continue

        for lineno, contexts in data.contexts_by_lineno(filename).items():
            seen_fail, seen_pass = set(), set()
            for c in contexts:
                kind = classify(c)
                if kind == "fail":
                    seen_fail.add(c.split("|")[0])
                elif kind == "pass":
                    seen_pass.add(c.split("|")[0])
            if seen_fail or seen_pass:
                cell = spectrum[(rel, lineno)]
                cell["ef"] += len(seen_fail)     # distinct tests, not events
                cell["ep"] += len(seen_pass)

    return spectrum


# =============================================== 5. SCORE + AGGREGATE + RANK
def rank_functions(spectrum, total_failing, project_root, top_n=5):
    """
    Score every line with Ochiai, roll the scores up to functions (MAX of the
    function's line scores - a function is as suspicious as its most suspicious
    line; a mean would dilute a real fault inside a long function), and rank.
    """
    by_file = defaultdict(list)
    for (rel_path, lineno), c in spectrum.items():
        score = ochiai(c["ef"], c["ep"], total_failing)
        if score > 0:
            by_file[rel_path].append((lineno, score))

    func_scores = {}
    for rel_path, lines in by_file.items():
        line_map = build_line_to_function_map(os.path.join(project_root, rel_path))
        for lineno, score in lines:
            fname = line_map.get(lineno)
            if not fname:
                continue                         # module-level code, not in a function
            key = (rel_path, fname)
            if score > func_scores.get(key, 0.0):
                func_scores[key] = score

    ranked = [
        {"file": f, "function": fn, "score": round(s, 3)}
        for (f, fn), s in sorted(func_scores.items(), key=lambda kv: -kv[1])
    ]
    return ranked[:top_n]


def render_for_prompt(ranked):
    """
    Format the ranking for the grounded critique prompt. Keep it to ~5 rows: a long
    list makes a 7B model grab row 1 without thinking.
    """
    return "\n".join(f"{r['file']} :: {r['function']}    {r['score']}" for r in ranked)


# ================================================================== SELF-TEST
# Needs no repo or environment: checks the formula, the ast mapping, and the
# test-name matcher, including the case that broke the sketch.
if __name__ == "__main__":
    assert ochiai(ef=1, ep=0, total_failing=1) == 1.0
    assert abs(ochiai(1, 9, 1) - 0.3162) < 1e-3
    assert ochiai(0, 5, 2) == 0.0
    assert abs(ochiai(2, 2, 2) - 0.7071) < 1e-3
    print("ochiai OK")

    m = build_line_to_function_map(__file__)
    assert any(v == "ochiai" for v in m.values())
    print("ast line->function mapping OK")

    # the sketch's bug: test_add must NOT match test_add_user
    assert same_test("test_calc.test_add", "test_add")
    assert not same_test("test_calc.test_add_user", "test_add")
    assert not same_test("test_calc.test_add", "test_add_user")
    # identifier formats from different sources agree
    assert same_test("tests.test_x.TestC.test_m|run", "tests/test_x.py::TestC::test_m")
    assert same_test("tests.test_x.TestC.test_m", "tests/test_x.py::TestC::test_m[case1]")
    assert same_test("tests.test_pysnooper.test_file_output",
                     "tests.test_pysnooper.test_file_output")
    assert not same_test("tests.test_x.TestA.test_m", "tests/test_x.py::TestB::test_m")
    print("test-name matching OK")

    assert _is_project_source("pysnooper/pysnooper.py")
    assert _is_project_source("tester/core.py")           # the sketch skipped this
    assert not _is_project_source("tests/test_calc.py")
    assert not _is_project_source("src\\tests\\helpers.py")  # Windows path
    assert not _is_project_source("test_calc.py")
    assert not _is_project_source("pkg/conftest.py")
    print("project-source filter OK")
