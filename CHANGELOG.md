# Changelog

Benchmark versions look like `vMAJOR.MINOR`. A new version starts whenever numbers stop being comparable: new items, a
label fix, a scorer change or a split change. Each version keeps its own `splits/` file and `results/` directory.
Package releases follow semver.

## v0.1 (package 0.1.0), 2026-11-01

This is the first release prepared for publication.

- It has 12 datasets with cards and deterministic builders, and the v0.1 split of 1,844 items in eval and fit halves.
- It has results for five systems that read text (Sage 2 rc9 fast, Clairic, NuExtract3, Schematron-3B and GLiNER2.5)
  and one that also reads the page image (Clairic with the image).
- The `extract-bench` CLI has the commands `datasets`, `run` (`ocr`, `read`, `score`, `export`), `report`, `validate`
  and `open` (`export` and `import` for GPU models). Runs are mock runs by default. A live call needs `EXB_LIVE=1` and
  a spend cap, and every call goes to a local ledger.
- `extract-bench validate` checks the submission format. Scripts in `scripts/` draw the README figures and the
  raw-output examples.

Before v0.1 we ran these internal reads, and this release folds them in:

- 2026-10-29: we added open-weight extraction models (NuExtract3, Schematron-3B, GLiNER2.5), run with vLLM 0.32 in
  bf16 at temperature 0 on one A40. Output that doesn't parse scores 0 on every field.
- 2026-10-28: Levanto read every item with Sage 2 rc9, fast tier, and fitted its abstention thresholds on the `fit`
  half. This read introduced the single-request item filter.
- 2026-10-26: we fixed the scorer for every system, and comments in `score.py` name each fix. The scorer now folds
  Unicode dashes and quotes, ignores a leading article and keeps digit runs apart when comparing text, compares money in
  cents rounded half up, leaves unparseable gold dates unscored, scores TAT-QA spans by token F1, and merges list
  answers across windows by their spans.
- 2026-10-23: we built and audited the datasets and made the first live reads of Clairic with text, with the image,
  and with both.
