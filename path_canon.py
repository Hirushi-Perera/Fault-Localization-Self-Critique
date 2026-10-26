"""
Map Agentless's predicted file paths back to real repository paths before scoring.

Why: Agentless's repo tree (get_repo_structure.create_structure) files every TOP-LEVEL file
under a made-up folder named after the repository. For psf/black the model is shown

    black/          <- not a real folder
        black.py    <- really the top-level file "black.py"
    blib2to3/ ...

so it (correctly, given what it saw) predicts "black/black.py", and Agentless accepts that
path because its own file list uses the same alias. Only our scorers, which compare against
the gold patch's real path "black.py", would count it as a miss. Found on the BugsInPy black
bugs (2026-10-05); it rarely shows on SWE-bench, whose code mostly lives inside package
folders, but the rule below is the same for every repo.

The rule changes a path ONLY when all three hold:
  1. it starts with "<repo folder>/" and the rest is a single file name (a top-level file);
  2. that path does not exist in the repository at the bug's commit;
  3. the path without the prefix does exist.
So real paths such as "tornado/web.py" (tornado's package folder is also called tornado) are
never touched, and no existing result whose paths resolved can change.

Agentless's own outputs are never modified - this is applied to a copy, inside the scorers.
"""
import contextlib
import io
import re

from get_repo_structure.get_repo_structure import repo_to_top_folder


def make_canon(repo, read):
    """canon(path) -> the real path for an Agentless-predicted path.
    `read` is a scorer's make_reader(repo, commit): source text, or None if missing."""
    top = repo_to_top_folder[repo]
    memo = {}

    def exists(path):
        with contextlib.redirect_stdout(io.StringIO()):     # silence 404 notices
            return read(path) is not None

    def canon(path):
        if not path:
            return path
        p = path.strip().replace("\\", "/")
        if p not in memo:
            out = p
            prefix = top + "/"
            if p.startswith(prefix):
                rest = p[len(prefix):]
                if "/" not in rest and not exists(p) and exists(rest):
                    out = rest
            memo[p] = out
        return memo[p]

    return canon


def _canon_keys(locs, canon):
    if isinstance(locs, dict):
        return {canon(k): v for k, v in locs.items()}
    if isinstance(locs, list):                    # num_samples > 1 shape
        return [_canon_keys(d, canon) for d in locs]
    return locs


def canon_patch(patch, canon):
    """Rewrite only the file-header lines of a diff (diff --git / --- a/ / +++ b/)."""
    out, in_header = [], False
    for line in (patch or "").splitlines(keepends=True):
        body = line.rstrip("\r\n")
        end = line[len(body):]
        m = re.match(r"^diff --git a/(\S+) b/(\S+)$", body)
        if m:
            body = f"diff --git a/{canon(m.group(1))} b/{canon(m.group(2))}"
            in_header = True
        elif in_header and body.startswith("@@"):
            in_header = False
        elif in_header:
            m = re.match(r"^(--- a/|\+\+\+ b/)(\S+)$", body)
            if m:
                body = m.group(1) + canon(m.group(2))
        out.append(body + end)
    return "".join(out)


def canon_record(record, canon):
    """A copy of an Agentless output record with every predicted path made real."""
    if not record:
        return record
    r = dict(record)
    if r.get("found_files"):
        r["found_files"] = [canon(p) for p in r["found_files"]]
    for key in ("found_related_locs", "found_edit_locs"):
        if key in r:
            r[key] = _canon_keys(r[key], canon)
    if r.get("model_patch"):
        r["model_patch"] = canon_patch(r["model_patch"], canon)
    return r
