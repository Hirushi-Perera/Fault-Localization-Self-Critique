"""
Score FUNCTION-level localization (Agentless stage 2) against the real developer's patch.

Stage 2 predicts classes and functions; the gold patch gives line numbers. Bridging the two
needs the buggy file's source, so each changed file is fetched at its `base_commit` from
GitHub raw and cached under results/.file_cache/ (gitignored). Roughly 20-30 small requests
the first time, nothing on re-runs.

Reuses the project's own verified machinery rather than reimplementing it:
    faithfulness_sketch.gold_functions()   patch + source -> {(file, function)}
    faithfulness_sketch.match_location()   tiered exact / file+tail / func_only matching

Using the SAME matcher the real experiment will use matters: these numbers are then directly
comparable to the faithfulness results later, rather than being scored by a one-off rule.

Usage:
    python score_function_localization.py
    python score_function_localization.py --results_dir results/pilot_full
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from datasets import load_dataset

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "capstone")))
from faithfulness_sketch import gold_functions, match_location  # noqa: E402

CACHE = os.path.join("results", ".file_cache")
ENTRY_RE = re.compile(r"^\s*(class|function):\s*(\S+)", re.MULTILINE)


def normalise(path):
    return (path or "").strip().replace("\\", "/").lstrip("./").lower()


def make_reader(repo, commit):
    """read_buggy_file(path) -> source at base_commit, or None. Cached on disk."""
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


def predicted_elements(record):
    """
    found_related_locs -> ordered [(file, name, kind)], best guess first.

    Stage 2 emits lines like 'class: Foo' and 'function: Foo.bar'. Both kinds are kept:
    a class-only prediction cannot match a gold (file, function) pair exactly, but it is
    not nothing either, so it is scored separately rather than silently dropped.
    """
    locs = record.get("found_related_locs") or {}
    if isinstance(locs, list):
        locs = locs[0] if locs else {}
    out = []
    for path, blocks in locs.items():
        if isinstance(blocks, str):
            blocks = [blocks]
        for block in blocks or []:
            for kind, name in ENTRY_RE.findall(block or ""):
                out.append((normalise(path), name, kind))
    return out


def class_level_hit(pred_file, pred_name, gold):
    """
    Did we name the CLASS that contains a gold function, in the right file?

    Gold functions are qualified ('ManyToManyRel.identity'), so a stage-2 answer of
    'class: ManyToManyRel' is a genuine partial hit -- the right container, without
    the specific method. Reported separately, never folded into the headline.
    """
    for gfile, gfunc in gold:
        gf = normalise(gfile)
        same_file = (gf == pred_file or gf.endswith(pred_file)
                     or pred_file.endswith(gf))
        if same_file and "." in gfunc and gfunc.split(".")[0] == pred_name:
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results/pilot_full_v2")
    parser.add_argument("--dataset", default="princeton-nlp/SWE-bench_Verified")
    args = parser.parse_args()

    print(f"Loading {args.dataset} ...")
    ds = load_dataset(args.dataset, split="test")
    meta = {r["instance_id"]: (r["repo"], r["base_commit"], r["patch"]) for r in ds}

    bugs = sorted(
        d for d in os.listdir(args.results_dir)
        if os.path.isfile(os.path.join(args.results_dir, d,
                                       "related", "loc_outputs.jsonl"))
    )

    rows = []
    for bug in bugs:
        path = os.path.join(args.results_dir, bug, "related", "loc_outputs.jsonl")
        with open(path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        repo, commit, patch = meta.get(bug, (None, None, None))
        if not repo:
            print(f"  ! {bug} not in dataset, skipping")
            continue

        print(f"  {bug} ...")
        gold = gold_functions(patch, make_reader(repo, commit))
        preds = predicted_elements(record)

        # best tier achieved, scanning predictions in rank order
        tier, rank, class_rank = None, None, None
        for i, (pfile, pname, kind) in enumerate(preds, start=1):
            if kind == "function":
                hit, t = match_location({"file": pfile, "function": pname}, gold)
                if hit and tier is None:
                    tier, rank = t, i
            elif kind == "class" and class_rank is None:
                if class_level_hit(pfile, pname, gold):
                    class_rank = i

        rows.append({
            "bug": bug, "n_pred": len(preds), "n_gold": len(gold),
            "tier": tier, "rank": rank, "class_rank": class_rank,
        })

    # ---------------- per-bug ----------------
    header = (f"{'bug':<34} {'preds':>5} {'gold':>4} {'tier':<11} {'rank':>4} "
              f"{'class-only':>10}")
    print("\n" + header)
    print("-" * len(header))
    for r in rows:
        tier = r["tier"] or "-"
        rank = r["rank"] if r["rank"] else "-"
        cls = f"rank {r['class_rank']}" if r["class_rank"] else "-"
        print(f"{r['bug']:<34} {r['n_pred']:>5} {r['n_gold']:>4} {tier:<11} "
              f"{str(rank):>4} {cls:>10}")
    print("-" * len(header))

    # ---------------- aggregate ----------------
    n = len(rows) or 1
    strict = [r for r in rows if r["tier"] in ("exact", "file+tail")]
    loose = [r for r in rows if r["tier"] == "func_only"]
    cls_only = [r for r in rows if r["tier"] is None and r["class_rank"]]

    print(f"\nn = {len(rows)} bugs\n")
    print("FUNCTION-LEVEL, scored with the project's own tiered matcher:")
    print(f"  exact (file + qualified function) : "
          f"{sum(1 for r in rows if r['tier'] == 'exact')}/{len(rows)}")
    print(f"  file + function short name        : "
          f"{sum(1 for r in rows if r['tier'] == 'file+tail')}/{len(rows)}")
    print(f"  -> acc@any (strict, the headline) : {len(strict)}/{len(rows)}"
          f"  ({len(strict) / n * 100:.1f}%)")
    print(f"  function name only, WRONG file    : {len(loose)}/{len(rows)}"
          f"   [loose bound, reported separately]")
    print(f"  right CLASS only, no function     : {len(cls_only)}/{len(rows)}"
          f"   [partial credit, not in the headline]")

    for k in (1, 3):
        hits = sum(1 for r in strict if r["rank"] and r["rank"] <= k)
        print(f"  acc@{k}: {hits}/{len(rows)} ({hits / n * 100:.1f}%)")

    ranks = [r["rank"] for r in strict if r["rank"]]
    if ranks:
        print(f"  MFR (strict hits only): {sum(ranks) / len(ranks):.3f}"
              f"   [over {len(ranks)}/{len(rows)} bugs]")

    print("\n  Stage 2 answers with classes as well as functions, so a class-only")
    print("  answer is a real partial result -- the right container without the")
    print("  method. Counted separately rather than dropped or folded in.")


if __name__ == "__main__":
    main()
