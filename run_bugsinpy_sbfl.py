"""
Run SBFL/Ochiai on a REAL bug from BugsInPy and check whether it ranks the function
the real developer's fix actually changed.

This is the step up from `examples/toy_sbfl/`: same mechanism (agentless/critique/sbfl.py), but the bug is real, the tests are
the project's own, and the gold location comes from the developer's patch rather than
from us planting it.

No Docker. No LLM. No API cost. BugsInPy ships each bug's metadata (buggy commit,
triggering test, real patch), so we clone the project at the buggy commit and run its
own test suite under coverage.

Usage (after cloning the project at the buggy commit):

    python run_bugsinpy_sbfl.py ^
        --repo "C:/Users/USER/Documents/IIT/4thYear/FYP/bugsinpy_work/PySnooper" ^
        --bug-dir "C:/Users/USER/Documents/IIT/4thYear/FYP/BugsInPy/projects/PySnooper/bugs/3"

What it does:
    1. runs the project's test suite under coverage with per-test contexts
    2. reads which tests passed and which failed (from pytest's JUnit XML)
    3. builds the spectrum and ranks functions by Ochiai
    4. extracts the gold location from the real patch
    5. reports WHERE the gold function landed in the ranking
"""
import argparse
import os
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))   # repo root

from agentless.critique.sbfl import (  # noqa: E402
    _is_project_source, build_line_to_function_map, build_spectrum, rank_functions,
    same_test,
)

COVERAGERC = "[run]\ndynamic_context = test_function\n"
GOLD_FILE_RE = re.compile(r"^diff --git a/(\S+)", re.MULTILINE)
HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+", re.MULTILINE)


def read_bug_info(bug_dir):
    """Parse bug.info's key="value" lines into a dict."""
    info = {}
    with open(os.path.join(bug_dir, "bug.info"), encoding="utf-8") as f:
        for line in f:
            if "=" in line:
                key, _, value = line.partition("=")
                info[key.strip()] = value.strip().strip('"').strip("'")
    return info


def prepare_checkout(repo, info):
    """
    Put the repo into the state BugsInPy intends: BUGGY source, but the test file
    from the FIXED commit.

    This is not optional. The test that catches the bug is normally ADDED as part of
    the fix, so it does not exist at the buggy commit -- check out the buggy commit
    alone and the suite passes cleanly, with nothing for SBFL to work from. Same
    protocol as SWE-bench's FAIL_TO_PASS tests and Defects4J.
    """
    buggy = info.get("buggy_commit_id")
    fixed = info.get("fixed_commit_id")
    test_file = info.get("test_file")
    if not (buggy and fixed and test_file):
        sys.exit(f"bug.info is missing required fields: {info}")

    print(f"Checking out buggy source   : {buggy[:12]}")
    subprocess.run(["git", "-C", repo, "checkout", "-q", "--force", buggy], check=True)

    # bug.info may list several files separated by ';' (e.g. a test plus its data file),
    # but the list can be incomplete: black bug 6 omits two data files its own triggering
    # tests need (one crashed with FileNotFoundError, which Ochiai then counted as a
    # failing test). So also take every TEST-side file the fix commit changed - the same
    # thing SWE-bench does by applying the fix's whole test patch. Source files are never
    # taken from the fixed commit.
    listed = [t.strip() for t in test_file.split(";") if t.strip()]
    changed = subprocess.run(
        ["git", "-C", repo, "diff", "--name-only", buggy, fixed],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    for tf in dict.fromkeys(listed + [c for c in changed if _is_test_path(c)]):
        src = "bug.info" if tf in listed else "fix diff"
        print(f"Overlaying test from fix    : {tf} @ {fixed[:12]}  ({src})")
        exists = subprocess.run(["git", "-C", repo, "cat-file", "-e", f"{fixed}:{tf}"],
                                capture_output=True).returncode == 0
        if exists:
            subprocess.run(["git", "-C", repo, "checkout", fixed, "--", tf], check=True)
        else:
            print(f"   (deleted by the fix - leaving the buggy version)")
    print()


def _is_test_path(rel):
    """A test file or test data file (anything under a test folder, or test-named)."""
    parts = rel.replace("\\", "/").split("/")
    name = parts[-1]
    return (any(p in ("test", "tests", "testing") for p in parts[:-1])
            or name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def run_tests(repo, tests_path="tests", ignore=()):
    """Run the suite under coverage, return (junit_path, coverage_path).
    `tests_path` is where the project keeps its tests (PySnooper: tests,
    tornado: tornado/test, youtube-dl: test). `ignore` lists test modules that
    cannot even be loaded on the bug's Python version -- skipped EXPLICITLY and
    printed, never silently."""
    rc = os.path.join(repo, ".coveragerc")
    with open(rc, "w", encoding="utf-8") as f:
        f.write(COVERAGERC)

    junit = os.path.join(repo, "junit.xml")
    for stale in (junit, os.path.join(repo, ".coverage")):
        if os.path.exists(stale):
            os.remove(stale)

    print("Running the project's own test suite under coverage ...")
    print("  (test failures here are EXPECTED - that is the bug)\n")
    subprocess.run(
        [sys.executable, "-m", "coverage", "run", "--rcfile=.coveragerc",
         "-m", "pytest", "-q", "--junitxml=junit.xml", tests_path]
        + [f"--ignore={p}" for p in ignore],
        cwd=repo,
        check=False,          # a failing test is the point, not an error
    )
    return junit, os.path.join(repo, ".coverage")


def split_tests(junit_path):
    """
    Read pytest's JUnit XML -> (failing, passing, collection_errors).

    A COLLECTION error (a test module that would not even import, usually a missing
    dependency) is NOT a failing test and must never be treated as one -- otherwise a
    broken environment gets silently recorded as a localization miss, which would
    quietly corrupt results across a batch run.
    """
    tree = ET.parse(junit_path)
    failing, passing, collect_errors = [], [], []
    for case in tree.iter("testcase"):
        # full dotted name (module + class + function), not just the function: two test
        # files can each define a test with the same function name
        cls, fn = case.get("classname") or "", case.get("name") or ""
        name = f"{cls}.{fn}" if cls else fn
        error = case.find("error")
        if error is not None:
            msg = (error.get("message") or "") + (error.text or "")
            # pytest records a module it could not even load as an <error> with an
            # empty classname and message "collection failure" (import error,
            # SyntaxError, ...); the other checks catch older pytest wording
            if (not cls or "collection failure" in msg or "collecting" in msg
                    or "ImportError" in msg or "ModuleNotFound" in msg):
                collect_errors.append(name)
                continue
            failing.append(name)
        elif case.find("failure") is not None:
            failing.append(name)
        else:
            passing.append(name)
    return failing, passing, collect_errors


def trigger_tests(bug_dir):
    """
    The bug-triggering test(s) BugsInPy names in run_test.sh, e.g.
        pytest -q -s tests/test_chinese.py::test_chinese
        python -m unittest -q test.test_utils.TestUtil.test_strip_jsonp
    """
    path = os.path.join(bug_dir, "run_test.sh")
    if not os.path.exists(path):
        return []
    tests = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            words = [w for w in line.split() if not w.startswith("-")]
            if words[:1] == ["pytest"]:
                tests += words[1:]
            elif words[:3] == ["python", "-m", "unittest"] or words[:2] == ["python", "unittest"]:
                tests += [w for w in words if "." in w and w != "unittest"]
    return tests


def gold_edit_lines(bug_dir):
    """
    {source file: set(buggy-file line numbers)} from the real patch, same convention
    as the pilot's FUNCTION-level scorer (faithfulness_sketch.parse_patch_old_lines):
    a replacement is located at its removed lines; a pure insertion at the old lines
    on EITHER SIDE of the insertion point. Both sides matter: code appended to the end
    of a function sits before a blank line, so the line after alone would miss the
    function it was added to. Test files are dropped -- a test is not where the bug lives.
    (Not imported: the capstone sketch lives outside this repo, and the Agentless
    scorer module pulls in `datasets`, which the BugsInPy environments do not have.)
    """
    with open(os.path.join(bug_dir, "bug_patch.txt"), encoding="utf-8") as f:
        patch = f.read()

    out, cur, old, removed, adds = {}, None, None, set(), set()

    def end_block():
        if cur:
            if removed:
                out.setdefault(cur, set()).update(removed)
            elif adds:
                p = min(adds)                    # old line the insertion sits before
                out.setdefault(cur, set()).update({p - 1, p})
        removed.clear()
        adds.clear()

    for line in patch.splitlines():
        m = GOLD_FILE_RE.match(line)
        if m:
            end_block()
            cur = m.group(1) if _is_project_source(m.group(1)) else None
            old = None
        elif old is None and line.startswith(("--- ", "+++ ")):
            continue
        elif line.startswith("@@"):
            end_block()
            old = int(HUNK_RE.match(line).group(1))
        elif old is not None:
            if line.startswith("-"):
                removed.add(old)
                old += 1
            elif line.startswith("+"):
                adds.add(old)
            elif not line.startswith("\\"):
                end_block()
                old += 1
    end_block()
    return out


def gold_functions(bug_dir, repo):
    """
    Map the gold edit lines onto functions in the BUGGY source.
    Returns (set of (file, function), {file: [module-level lines]}). Module-level edits
    (imports, constants) belong to no function, so no function ranking can hit them.
    """
    funcs, module_level = set(), {}
    for rel, lines in gold_edit_lines(bug_dir).items():
        path = os.path.join(repo, rel)
        line_map = build_line_to_function_map(path)
        with open(path, encoding="utf-8", errors="ignore") as f:
            src = f.read().splitlines()
        for ln in sorted(lines):
            name = line_map.get(ln)
            if name:
                funcs.add((rel, name))
            elif 0 < ln <= len(src) and src[ln - 1].strip():
                # a blank anchor line (the gap after a function) is not an edit
                module_level.setdefault(rel, []).append(ln)
    return funcs, module_level


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="project checked out at the BUGGY commit")
    parser.add_argument("--bug-dir", required=True, help="BugsInPy projects/<p>/bugs/<n> folder")
    parser.add_argument("--top-n", type=int, default=10, help="rows to print")
    parser.add_argument("--tests", default="tests",
                        help="test folder to run, relative to --repo (tornado: tornado/test)")
    parser.add_argument("--ignore", action="append", default=[],
                        help="test module to skip (repeatable); only for modules that "
                             "cannot load on the bug's Python version")
    parser.add_argument("--failing", choices=["trigger", "all"], default="trigger",
                        help="trigger = only BugsInPy's triggering test(s) count as failing")
    args = parser.parse_args()

    repo = os.path.abspath(args.repo)
    info = read_bug_info(args.bug_dir)
    prepare_checkout(repo, info)
    if args.ignore:
        print(f"Skipping test modules that cannot load on this Python: {args.ignore}")
    junit, cov = run_tests(repo, args.tests, args.ignore)

    if not os.path.exists(junit):
        sys.exit("pytest produced no junit.xml - did the suite fail to start?")

    failing, passing, collect_errors = split_tests(junit)

    # Environment problems must abort, never fall through to a ranking. A missing
    # dependency is not a localization result and must not be recorded as one.
    if collect_errors:
        print("\nSETUP ERROR - test modules could not even be imported:")
        for name in collect_errors:
            print(f"  {name}")
        sys.exit(
            "\nThe test suite never ran, so there is nothing to localize. This is an\n"
            "environment problem, NOT a result - do not record it as a miss.\n"
            "Usually a missing dependency: install the project's own\n"
            "requirements.txt and test_requirements.txt, then re-run."
        )

    # Only BugsInPy's bug-triggering test(s) count as "failing". Any OTHER test that
    # fails on this machine is failing for a different reason (platform, locale,
    # pre-existing breakage) and says nothing about THIS bug; counting it as failing
    # would pull its code up the ranking. It is left out of the spectrum entirely --
    # it did not pass either, so it is not evidence of correct code.
    triggers = trigger_tests(args.bug_dir)
    unrelated = []
    if args.failing == "trigger" and triggers:
        unrelated = [t for t in failing if not any(same_test(t, g) for g in triggers)]
        failing = [t for t in failing if t not in unrelated]
    print(f"\ntriggering test(s) per BugsInPy: {triggers or 'none listed'}")
    print(f"failing tests used ({len(failing)}): {failing}")
    if unrelated:
        print(f"other failures EXCLUDED ({len(unrelated)}): {unrelated}")
    print(f"passing tests ({len(passing)}): {len(passing)} tests")
    if not failing:
        sys.exit("\nNo failing test - SBFL needs one. Wrong commit checked out?")
    if not passing:
        sys.exit(
            "\nNo passing tests. Ochiai needs both: with nothing passing, every line "
            "the failing test touched scores identically and the ranking is meaningless."
        )

    spectrum = build_spectrum(cov, failing_tests=failing,
                              passing_tests=passing, project_root=repo)
    # full ranking (top_n=None), so a gold function's rank is known even below the
    # printed rows; equal scores keep the order they are produced in (ordinal ties)
    ranked = rank_functions(spectrum, total_failing=len(failing),
                            project_root=repo, top_n=None)

    gold_funcs, module_level = gold_functions(args.bug_dir, repo)
    gold_files = {f for f, _ in gold_funcs} | set(module_level)

    def mark(e):
        if (e["file"], e["function"]) in gold_funcs:
            return "  <- GOLD FUNCTION"
        return "  (gold file)" if e["file"] in gold_files else ""

    print(f"\nOchiai ranking (top {args.top_n} of {len(ranked)} functions):")
    for i, e in enumerate(ranked[:args.top_n], start=1):
        print(f"  {i:3}. {e['file']} :: {e['function']:32s} {e['score']:.3f}{mark(e)}")

    print("\nThe real fix (source files only):")
    for f, fn in sorted(gold_funcs):
        print(f"  function      {f} :: {fn}")
    for f, lines in sorted(module_level.items()):
        print(f"  module-level  {f} lines {lines}  (outside any function)")

    def rank_and_ties(i):
        s = ranked[i]["score"]
        return i + 1, sum(1 for e in ranked if e["score"] == s) - 1

    print("\nWhere each gold function landed:")
    func_ranks = []
    for f, fn in sorted(gold_funcs):
        i = next((i for i, e in enumerate(ranked) if (e["file"], e["function"]) == (f, fn)), None)
        if i is None:
            print(f"  {fn:32s} NOT RANKED (score 0: no triggering test ran it)")
        else:
            r, ties = rank_and_ties(i)
            func_ranks.append(r)
            print(f"  {fn:32s} rank {r:3d} of {len(ranked)}, score {ranked[i]['score']:.3f}"
                  f"{f'  (tied with {ties} others)' if ties else ''}")

    file_rank = next((i + 1 for i, e in enumerate(ranked) if e["file"] in gold_files), None)
    func_rank = min(func_ranks) if func_ranks else None

    # ------------------------------------------------------------------
    # Published metrics only (supervisor rule): acc@k from AutoFL (Kang et al.,
    # FSE 2024) at k = 1, 3, 5, and the rank that DeepFL's MFR averages
    # (Li et al., ISSTA 2019). AutoFL's acc@k is method-level and counts a hit
    # when ANY gold method is in the top k -- so FUNCTION level is the headline.
    # The file-level row is kept only because it is what this script used to report.
    #
    # NOTE ON SCOPE: one bug gives one observation per metric, not a RATE.
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    print("METRICS (single bug -- one observation, not a rate)")
    print("=" * 62)
    for label, r in (("FUNCTION", func_rank), ("file", file_rank)):
        hits = "  ".join(f"acc@{k} {'HIT' if r and r <= k else 'miss'}" for k in (1, 3, 5))
        print(f"  {label:8s} first correct rank: {r or '-':>4}   {hits}")
    print("=" * 62)
    if not gold_funcs:
        print("NOTE: the fix touches no function (module-level only) -- function-level\n"
              "acc@k cannot hit on this bug by construction; report it, don't drop it.")


if __name__ == "__main__":
    main()
