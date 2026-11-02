"""extract-bench: command-line entry point.

  extract-bench datasets                     list datasets and item counts
  extract-bench run   ocr|read|score|export  the harness stages (mock by default; see `extract-bench run -h`)
  extract-bench report                       score results/<version>/ and print the leaderboard
  extract-bench validate DIR                 check a submission directory (format, coverage, re-score)
  extract-bench open  export|import          open-weight models: build GPU requests / import GPU outputs
"""
from __future__ import annotations

import sys

from . import __version__

COMMANDS = ("datasets", "run", "report", "validate", "open")


def _datasets(argv):
    from .common import BENCH_VERSION, list_datasets, load_items, load_split
    sp = load_split(BENCH_VERSION)
    print(f"{'dataset':<17} {'items':>6} {'v0.1 eval':>9} {'v0.1 fit':>8}  use case")
    for k in list_datasets():
        its = load_items(k)
        s = sp.get(k, {"eval": [], "fit": []})
        print(f"{k:<17} {len(its):>6} {len(s['eval']):>9} {len(s['fit']):>8}  {its[0].get('use_case', '')}")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__.strip()); return 0
    if argv[0] in ("-V", "--version"):
        print(f"extract-bench {__version__}"); return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "datasets":
        return _datasets(rest)
    if cmd == "run":
        from .run import main as run_main
        return run_main(rest)
    if cmd == "report":
        from .report import main as report_main
        return report_main(rest)
    if cmd == "validate":
        from .validate import main as validate_main
        return validate_main(rest)
    if cmd == "open":
        from .open_extractors import main as open_main
        return open_main(rest) or 0
    print(f"unknown command {cmd!r}; one of {', '.join(COMMANDS)}", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
