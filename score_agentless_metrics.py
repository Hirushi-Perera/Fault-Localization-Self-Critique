"""
Score localization and repair with the metrics the Agentless paper itself publishes.

Source: Xia et al., "Demystifying LLM-Based Software Engineering Agents" (Agentless),
FSE 2025 -- `xia2025agentless` in references.bib.

  CONTAINS GT  (Table 2, localization ablation)
      "the percentage of problems whose ground truth edit locations remain in the
       location set" -- reported after each localization step.
  LOC          (Table 2)
      "the average lines of code of each location set" -- the size cost of the location
       set, so containment cannot be bought by simply returning a huge window.
  % CORRECT LOCATION  (Section 4, Metrics)
      "the percent of problems where the patch produced by the tool covers the edit
       location(s) of the ground truth developer patch ... over three granularities:
       file, function, and line ... a patch contains the correct location if it edits
       a superset of all locations in the ground truth patch."

No metric here is invented. Where the paper's evaluation script was not released, this
reconstructs it from the authors' OWN functions rather than writing new rules:

  location set  <- agentless.util.preprocess_data.transfer_arb_locs_to_locs()
                   the exact function repair uses to decide which lines the model sees:
                   each `line: N` -> that line, each `function:` / `class:` -> its full
                   span, each widened by context_window, merged. Intervals inclusive.
  ground truth  <- the convention of agentless.util.preprocess_data.compile_gt_locations():
                   modified/removed lines, plus ONE insertion point per contiguous block
                   of added lines.

IMPORTANT - these are "ALL locations" metrics. A bug counts only if EVERY ground-truth
location is covered. That is deliberately stricter than acc@k (AutoFL), which counts a bug
if ANY one location is found. Report both; the gap between them is informative.

Reads results on disk only. No model calls. Buggy source files are fetched from GitHub at
each bug's base_commit and cached in results/.file_cache/ (shared with
score_function_localization.py).

Usage:
    python score_agentless_metrics.py
    python score_agentless_metrics.py --results_dir results/multifile
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from datasets import load_dataset

from agentless.util.preprocess_data import merge_intervals, transfer_arb_locs_to_locs
from patch_offsets import realign_patch

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "capstone")))
from faithfulness_sketch import _line_to_function_map_from_source  # noqa: E402

CACHE = os.path.join("results", ".file_cache")
HUNK_RE = re.compile(r"@@ -(\d+)(?:,\d+)? \+")


def norm(path):
    return (path or "").strip().replace("\\", "/").lstrip("./").lower()


# ----------------------------------------------------------------------------- source
def make_reader(repo, commit):
    """Same cache key format as score_function_localization.py, so fetches are shared."""
    def read(path):
        key = f"{repo.replace('/', '__')}__{commit[:12]}__{path.replace('/', '__')}"
        cached = os.path.join(CACHE, key)
        if os.path.isfile(cached):
            with open(cached, encoding="utf-8", errors="replace") as f:
                return f.read()
        url = f"https://raw.githubusercontent.com/{repo}/{commit}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                src = resp.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            print(f"    ! could not fetch {path}: {exc}")
            return None
        os.makedirs(CACHE, exist_ok=True)
        with open(cached, "w", encoding="utf-8") as f:
            f.write(src)
        return src
    return read


# ----------------------------------------------------------------------- ground truth
def edit_locations(patch):
    """
    {file: set(buggy-file line numbers)}, following compile_gt_locations(): removed
    lines are locations, and pure additions contribute ONE insertion point per
    contiguous run (merge_intervals, then the start).

    One interpretation had to be made, because the authors' code that splits a diff
    into `edits` is not released: a REPLACEMENT (a change block containing both '-'
    and '+' lines) is located at its removed lines only. Its '+' lines replace those
    lines rather than touching the line after them; counting that trailing position
    as an extra location would make every containment check stricter than the change
    actually is. A block with '+' lines and NO '-' lines is a pure insertion and
    contributes its insertion point.

    Applied identically to the gold patch and the model's patch.
    """
    out = {}
    cur, old = None, None
    blk_removed, blk_adds = set(), set()

    def end_block():
        # classify the change block just finished: replacement vs pure insertion
        target = out.setdefault(cur, {"removed": set(), "adds": set()})
        if blk_removed:
            target["removed"].update(blk_removed)
        else:
            target["adds"].update(blk_adds)
        blk_removed.clear()
        blk_adds.clear()

    for line in (patch or "").splitlines():
        if line.startswith("diff --git"):
            if cur is not None:
                end_block()
            cur, old = None, None
        # file headers appear only before a file's first hunk (old is None there);
        # inside a hunk, a removed line such as "-- sql comment" must NOT be read as
        # a "--- " header
        elif old is None and line.startswith("--- a/"):
            cur = line[6:].strip()
        elif old is None and line.startswith("--- "):
            cur = None                          # /dev/null: file created by the patch
        elif old is None and line.startswith("+++"):
            continue
        elif line.startswith("@@"):
            if cur is not None:
                end_block()
            m = HUNK_RE.match(line)
            old = int(m.group(1)) if m else None
        elif cur and old is not None:
            if line.startswith("-"):
                blk_removed.add(old)
                old += 1
            elif line.startswith("+"):
                blk_adds.add(old)
            elif line.startswith("\\"):
                continue
            else:                               # context line closes a change block
                if blk_removed or blk_adds:
                    end_block()
                old += 1
    if cur is not None:
        end_block()

    result = {}
    for f, parts in out.items():
        points = [st for st, _ in
                  merge_intervals(sorted((i, i + 1) for i in parts["adds"]))]
        lines = parts["removed"] | set(points)
        if lines:
            result[f] = lines
    return result


# ------------------------------------------------------------------------ location sets
def interval_set(file_to_locs, files, read, context_window):
    """
    {file: [(start, end), ...]} built with Agentless's own transfer_arb_locs_to_locs(),
    restricted to `files` -- mirroring repair, which only ever uses found_files[:top_n].
    """
    out = {}
    for path in files:
        locs = file_to_locs.get(path)
        if not locs:
            continue
        src = read(path)
        if src is None:
            continue
        try:
            line_locs, intervals = transfer_arb_locs_to_locs(
                locs, None, path, context_window=context_window,
                loc_interval=True, file_content=src)
        except Exception as exc:                        # never silently count a crash
            print(f"    ! could not build location set for {path}: {exc}")
            continue
        if line_locs:
            out[norm(path)] = intervals
    return out


def contains(gt, intervals):
    """Every ground-truth line, in every ground-truth file, inside the location set."""
    for gfile, lines in gt.items():
        spans = intervals.get(norm(gfile))
        if not spans:
            return False
        for ln in lines:
            if not any(st <= ln <= en for st, en in spans):
                return False
    return True


def loc_size(intervals):
    return sum(en - st + 1 for spans in intervals.values() for st, en in spans)


def as_dict(locs):
    if isinstance(locs, list):
        locs = locs[0] if locs else {}
    return locs or {}


def first_record(path):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


# -------------------------------------------------------------------- functions helper
def functions_of(file_lines, read):
    """{(file, function-or-None)} for every edited line; None = module level."""
    out = set()
    for path, lines in file_lines.items():
        src = read(path)
        fmap = _line_to_function_map_from_source(src) if src else {}
        for ln in lines:
            out.add((norm(path), fmap.get(ln)))
    return out


# ---------------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/pilot_full_v2")
    parser.add_argument("--stage1_fallback", default="results/pilot_rerun",
                        help="where to find stage-1 output if not in results_dir")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="test")
    meta = {r["instance_id"]: (r["repo"], r["base_commit"], r["patch"]) for r in ds}

    bugs = sorted(d for d in os.listdir(args.results_dir)
                  if os.path.isdir(os.path.join(args.results_dir, d))
                  and not d.startswith("."))

    rows = []
    for bug in bugs:
        if bug not in meta:
            continue
        repo, commit, gold_patch = meta[bug]
        read = make_reader(repo, commit)
        print(f"  {bug} ...")
        base = os.path.join(args.results_dir, bug)

        # repair's own settings, so the line-level location set is exactly what it saw
        rargs_path = os.path.join(base, "repair", "args.json")
        rargs = {}
        if os.path.isfile(rargs_path):             # pretty-printed JSON, not JSONL
            with open(rargs_path, encoding="utf-8") as f:
                rargs = json.load(f)
        top_n = rargs.get("top_n", 3)
        ctx = rargs.get("context_window", 25)

        s1 = (first_record(os.path.join(base, "loc_outputs.jsonl"))
              or first_record(os.path.join(args.stage1_fallback, bug, "loc_outputs.jsonl")))
        s2 = first_record(os.path.join(base, "related", "loc_outputs.jsonl"))
        s3 = (first_record(os.path.join(base, "merged", "loc_merged_0-0_outputs.jsonl"))
              or first_record(os.path.join(base, "edit_loc", "loc_outputs.jsonl")))
        rp = first_record(os.path.join(base, "repair", "output_0_processed.jsonl"))

        # gold hunk headers can be offset from where the patch really applies
        # (sklearn-10908: 22 lines) - locate each hunk in the buggy file first
        gold_patch, report = realign_patch(gold_patch, read)
        for f, hdr, actual in report:
            if actual is None:
                print(f"    ! gold hunk in {f} (header line {hdr}) not found in source")
            elif actual != hdr:
                print(f"    gold hunk in {f} realigned: header {hdr} -> actual {actual}")
        gt = edit_locations(gold_patch)
        gt_files = {norm(f) for f in gt}
        found = (s1 or {}).get("found_files") or []
        files = found[:top_n]

        # -- CONTAINS GT + LoC, per localization step --
        # file level: the location set is the whole of each found file
        file_ok = gt_files <= {norm(f) for f in files}
        file_loc = 0
        for f in files:
            src = read(f)
            file_loc += len(src.splitlines()) if src else 0

        # function level: spans of the classes/functions stage 2 named (no widening)
        fn_iv = interval_set(as_dict((s2 or {}).get("found_related_locs")), files, read, 0)
        # line level: exactly what repair saw -- widened by repair's context_window
        ln_iv = interval_set(as_dict((s3 or {}).get("found_edit_locs")), files, read, ctx)

        # -- % CORRECT LOCATION, on the produced patch --
        patch = ((rp or {}).get("model_patch") or "").strip()
        pl = edit_locations(patch) if patch else {}
        p_files = {norm(f) for f in pl}
        c_file = bool(patch) and gt_files <= p_files
        c_line = bool(patch) and all(
            lines <= pl.get(f, set()) for f, lines in gt.items())
        c_func = bool(patch) and functions_of(gt, read) <= functions_of(pl, read)

        rows.append({
            "bug": bug, "n_gt_files": len(gt_files),
            "file": file_ok, "file_loc": file_loc,
            "func": contains(gt, fn_iv), "func_loc": loc_size(fn_iv),
            "line": contains(gt, ln_iv), "line_loc": loc_size(ln_iv),
            "patch": bool(patch),
            "c_file": c_file, "c_func": c_func, "c_line": c_line,
        })

    if not rows:
        sys.exit("nothing scored - check --results_dir")

    yn = lambda b: "yes" if b else "-"
    n = len(rows)
    pct = lambda k: sum(1 for r in rows if r[k]) / n * 100
    avg = lambda k: sum(r[k] for r in rows) / n

    print("\nCONTAINS GT (Agentless Table 2) - every ground-truth location in the set")
    h = f"{'bug':<34} {'gt files':>8} {'file':>5} {'func':>5} {'line':>5} {'line LoC':>9}"
    print(h)
    print("-" * len(h))
    for r in rows:
        print(f"{r['bug']:<34} {r['n_gt_files']:>8} {yn(r['file']):>5} "
              f"{yn(r['func']):>5} {yn(r['line']):>5} {r['line_loc']:>9}")
    print("-" * len(h))
    print(f"  file      : {pct('file'):5.1f}%   avg LoC {avg('file_loc'):8.0f}")
    print(f"  function  : {pct('func'):5.1f}%   avg LoC {avg('func_loc'):8.0f}")
    print(f"  line      : {pct('line'):5.1f}%   avg LoC {avg('line_loc'):8.0f}")

    print("\n% CORRECT LOCATION (Agentless Sec. 4) - patch edits a superset of all GT")
    h = f"{'bug':<34} {'patch':>6} {'file':>5} {'func':>5} {'line':>5}"
    print(h)
    print("-" * len(h))
    for r in rows:
        print(f"{r['bug']:<34} {yn(r['patch']):>6} {yn(r['c_file']):>5} "
              f"{yn(r['c_func']):>5} {yn(r['c_line']):>5}")
    print("-" * len(h))
    print(f"  file      : {pct('c_file'):5.1f}%")
    print(f"  function  : {pct('c_func'):5.1f}%")
    print(f"  line      : {pct('c_line'):5.1f}%")
    print(f"\n  n = {n}; denominators are ALL bugs, as in the paper - an empty patch")
    print("  counts as not containing the correct location.")


if __name__ == "__main__":
    main()
