"""
Phase 1 check on a controlled example: does SBFL / Ochiai rank the planted bug first?

Toy repo: calc.py has three functions, one with a planted bug (`percentage` multiplies by
10 instead of 100). test_calc.py has 3 tests: 2 pass, 1 fails.

Run from this folder:

    coverage run --rcfile=.coveragerc -m pytest
    python run_sbfl.py

Expected: `percentage` ranks first with score 1.0.
    ef = 1 (the one failing test executes it), ep = 0 (no passing test does)
    Ochiai = 1 / sqrt(1 * (1 + 0)) = 1.0
`add` and `subtract` score 0 - no failing test executes them - so they are not listed.

Uses the project module agentless/critique/sbfl.py, not a copy.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))   # repo root

from agentless.critique.sbfl import build_spectrum, rank_functions  # noqa: E402

FAILING = ["test_percentage"]
PASSING = ["test_add", "test_subtract"]

os.chdir(HERE)
if not os.path.exists(".coverage"):
    sys.exit("No .coverage file. Run first:  coverage run --rcfile=.coveragerc -m pytest")

# show what coverage recorded, so a failure explains itself
from coverage import CoverageData  # noqa: E402
data = CoverageData(".coverage")
data.read()
contexts = sorted(c for c in data.measured_contexts() if c)
print("contexts recorded by coverage:")
for c in contexts:
    print(f"   {c}")
if not contexts:
    print("   (none - was --rcfile=.coveragerc passed?)")

spectrum = build_spectrum(".coverage", FAILING, PASSING, project_root=HERE)

print("\nspectrum (ef = failing tests that ran the line, ep = passing tests):")
for (f, ln), c in sorted(spectrum.items()):
    print(f"   {f}:{ln:<3}  ef={c['ef']}  ep={c['ep']}")

ranked = rank_functions(spectrum, total_failing=len(FAILING), project_root=HERE, top_n=5)
print("\nOchiai ranking:")
for i, r in enumerate(ranked, start=1):
    print(f"   {i}. {r['file']} :: {r['function']}   score {r['score']}")

print()
if ranked and ranked[0]["function"] == "percentage" and ranked[0]["score"] == 1.0:
    print("PASS: the planted bug ranks first with the maximum score (1.0).")
else:
    print(f"FAIL: expected percentage first at 1.0, got {ranked[:1]}")
    sys.exit(1)
