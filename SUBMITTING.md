# Submitting a system

A submission is a pull request that adds one directory, `results/<version>/<system>/`. CI re-scores the predictions you
submit, so your leaderboard number always comes from your files and the public scorer.

## 1. Files

```
results/v0.1/<system>/
  metadata.json
  preds/<dataset>.jsonl     one file per dataset, one JSON object per line
```

`<system>` is a short lowercase name such as `acme-extract-2`.

`metadata.json` looks like this:

```json
{
  "system": "acme-extract-2",
  "display_name": "Acme Extract 2",
  "org": "Acme",
  "model": "acme-extract-2-2026-11-01",
  "benchmark_version": "v0.1",
  "input": "text",
  "run_date": "2026-11-01",
  "open_weights": false,
  "maintainer_product": false,
  "tier": "self-reported",
  "notes": "Default settings; one request per document; no prompt tuning on eval items.",
  "report_url": "https://…",
  "contact": "…"
}
```

Set `input` to `text` for the main board; those systems read the text that `items.jsonl` and the OCR caches provide.
Use `image` or `image+text` for the image board. `model` is the exact public model or version identifier you ran. Use
`notes` for anything a reader needs to reproduce the run, including any tuning and the split you tuned on.

Each `preds/<dataset>.jsonl` has one line for every item of that dataset in `splits/v0.1.json`, from both halves. A
line holds either a prediction or an error:

```json
{"id": "fatura-T10-I6", "dataset": "fatura", "preds": {"invoice_number": {"value": "INV/55-24/340", "conf": 0.98}, "po_number": {"value": null, "conf": 0.91}}}
{"id": "fatura-T10-I7", "dataset": "fatura", "error": "HTTP 500 after 4 retries"}
```

`preds` has one entry per field name of the item. `value` is your system's answer, and `null` means the document
doesn't contain the field. Give numbers and money as JSON numbers, dates as `YYYY-MM-DD`, times as `HH:MM[:SS]`,
durations as ISO 8601 (`P2Y`), and list fields as JSON arrays. The scorer also parses written forms
([docs/METHODOLOGY.md](docs/METHODOLOGY.md) §4), but typed values are the safest choice. `conf` is optional, and only
the secondary rule (b) uses it.

Please add `wall_ms`, `n_calls`, `usage` (tokens) and `cost_usd` to each line when you have them. We keep them for a
future cost and latency board.

If your system ran but its output didn't parse, write `"preds": {}` with `"unparsed": true`. Every field then scores 0.
Don't drop the line.

## 2. Check it

```sh
extract-bench validate results/v0.1/<system>      # checks format and coverage, prints eval accuracy per dataset
extract-bench report                              # shows your row on the full board and rewrites board.json / board.md
```

## 3. Pull request checklist

Copy this into the pull request description:

- [ ] `extract-bench validate` passes without `--partial`.
- [ ] The run reads only the inputs the benchmark provides. It makes no web lookups and doesn't use `gold` fields or other systems' predictions.
- [ ] The system made one attempt per item. You didn't pick the best of several runs by checking them against the gold.
- [ ] Any tuning of prompts or thresholds used the `fit` half only, and `notes` says what you tuned.
- [ ] You declare, per dataset, whether the model's training data includes any of the 12 source datasets (yes, no or unknown).
- [ ] You declare any commercial interest in the benchmark or in another listed system.
- [ ] The files contain no API keys, internal URLs or private prompts.

## 4. What happens next

1. CI runs `validate` and `report`, and the pull request shows how the board changes.
2. A maintainer reviews and merges it, and the row appears as self-reported.
3. If you share an endpoint or weights and a maintainer re-runs your system, the row becomes reproduced. For an
   expensive system, the maintainer may re-run a random subset.
4. We never remove a row silently. A correction is a new commit with a line in [CHANGELOG.md](CHANGELOG.md).

Every run of a system that we publish stays published. We don't keep the best of several private attempts.
