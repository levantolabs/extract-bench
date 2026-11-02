"""Write scripts/realdoc_official.py: verbatim copies of the RealDoc-Bench template builder and scorer.

Reads the upstream files as TEXT (never imports or executes them) from
raw/github/src/realdoc-bench-<sha>/realdoc_bench/evaluate/{jsonify,score}.py at commit
fb26a6876481de76dc293f722ab4efa71279904d (Apache-2.0), pulls the named top-level definitions out with
`ast.get_source_segment`, and writes them, unchanged and in upstream order, into one module. The only
non-verbatim lines are the header and the guarded rapidfuzz import (so the build runs without rapidfuzz;
fuzzy_equal needs it at scoring time).

Usage: uv run --no-project python -I datasets/realdoc/scripts/vendor_official.py
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

HERE = Path(__file__).resolve().parent
DS = HERE.parent
SHA = "fb26a6876481de76dc293f722ab4efa71279904d"
SRC = DS / "raw" / "github" / "src" / f"realdoc-bench-{SHA}" / "realdoc_bench" / "evaluate"
OUT = HERE / "realdoc_official.py"

WANT = {
    "jsonify.py": ["_INT_LIKE", "_NUMBER_LIKE", "_DATE_LIKE", "_STRING_LIKE", "_BOOL_LIKE", "_BOOL_PAIRS",
                   "_placeholder_to_type_hint", "render_template", "_DATE_FORMATS", "_to_iso_date",
                   "widen_types_from_gold"],
    "score.py": ["DEFAULT_MODEL", "FUZZ_THRESHOLD", "FUZZ_MIN_WORDS", "SYSTEM_PROMPT", "_WS_AROUND_SEP",
                 "_WS_RUN", "_DASH_RE", "_SMART_QUOTES", "_TAG_RE", "_STYLE_RE", "_norm_str", "deep_equal",
                 "fuzzy_equal", "_TYPE_RE", "string_keys", "score_typed", "_RF_PREFIX", "_PAIR_RE",
                 "build_template"],
}


def names_of(node) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def main() -> None:
    parts = []
    for fname, want in WANT.items():
        src = (SRC / fname).read_text()
        tree = ast.parse(src)
        got = {}
        for node in tree.body:
            for n in names_of(node):
                if n in want:
                    got[n] = ast.get_source_segment(src, node)
        missing = [n for n in want if n not in got]
        if missing:
            raise SystemExit(f"{fname}: missing {missing}")
        sha = hashlib.sha256(src.encode()).hexdigest()
        parts.append(f"# ---- verbatim from realdoc_bench/evaluate/{fname} (sha256 {sha})\n")
        for n in want:
            seg = got[n]
            assert seg in src
            parts.append(seg + "\n\n")
    header = (
        '"""VERBATIM copies of RealDoc-Bench template building + scoring (generated; do not edit by hand).\n\n'
        f"Source: https://github.com/extend-hq/realdoc-bench at commit {SHA} (Apache-2.0).\n"
        "Regenerate with scripts/vendor_official.py; check with `verify()` below. Only the header, the guarded\n"
        "rapidfuzz import and verify() are ours. fuzzy_equal() needs rapidfuzz (scoring time only).\n"
        '"""\n'
        "from __future__ import annotations\n\n"
        "import json\nimport re\nfrom datetime import datetime\nfrom typing import Any\n\n"
        "try:\n    from rapidfuzz import fuzz\nexcept ImportError:  # build-time only; fuzzy_equal needs it\n"
        "    fuzz = None\n\n"
    )
    footer = (
        "def verify(upstream_dir) -> int:\n"
        '    """Re-check that every vendored definition is a verbatim substring of the upstream files."""\n'
        "    import ast as _ast, inspect as _inspect, pathlib as _pl, sys as _sys\n"
        "    me = _pl.Path(__file__).read_text()\n"
        "    up = ''.join((_pl.Path(upstream_dir) / f).read_text() for f in ('jsonify.py', 'score.py'))\n"
        "    n = 0\n"
        "    for node in _ast.parse(me).body:\n"
        "        if isinstance(node, (_ast.FunctionDef, _ast.Assign)) and getattr(node, 'name', '') != 'verify':\n"
        "            seg = _ast.get_source_segment(me, node)\n"
        "            if seg.startswith('fuzz = None'):\n"
        "                continue\n"
        "            if seg not in up:\n"
        "                raise AssertionError('not verbatim: ' + seg[:80])\n"
        "            n += 1\n"
        "    return n\n"
    )
    OUT.write_text(header + "".join(parts) + footer)
    print("wrote", OUT, "defs", sum(len(v) for v in WANT.values()))


if __name__ == "__main__":
    main()
