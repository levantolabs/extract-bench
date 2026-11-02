# Contributing

## Setup

```sh
uv venv && uv pip install -e ".[dev]" && source .venv/bin/activate
pytest -q                     # runs offline; skips the Tesseract checks if Tesseract isn't installed
```

## What we welcome

- New systems. See [SUBMITTING.md](SUBMITTING.md), or add an adapter as [docs/ADAPTERS.md](docs/ADAPTERS.md) describes.
- Label errors. Open an issue with the dataset, item id, field, current gold value and your evidence, such as a page
  crop or a quote. We batch fixes into the next benchmark version and list each one by item id in
  [CHANGELOG.md](CHANGELOG.md). Old versions stay reproducible.
- Scorer bugs, with a failing test. A scorer change starts a new benchmark version and re-scores every system.
- Unfair adapters. If a listed system's adapter misuses its API, tell us (see [CONFLICTS.md](CONFLICTS.md)).

## Rules for changes

- Don't edit numbers by hand. `extract-bench report` writes every result to `results/<version>/board.*`, and the
  scripts in `scripts/` draw every figure.
- Don't edit `items.jsonl` by hand. Change the dataset's `build.py` and rebuild. Gold corrections go through a
  versioned errata step.
- Keep every call to a paid API behind `require_live()` and `_ledger()`, so mock runs never touch the network.
- Treat files under `datasets/*/raw/` as untrusted, because rebuilds download them. Run Python with `-I` and never
  execute anything from `raw/`.
