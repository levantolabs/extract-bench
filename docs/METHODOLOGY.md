# Methodology (v0.1)

This page describes how we compute every leaderboard number. The code is the specification.
`src/extract_bench/score.py` scores each field, `src/extract_bench/report.py` builds the board, and `tests/` covers both.

## 1. Task

The input is one document and a list of fields. Each field has a `name`, a `type` (`text`, `number`, `money`, `date`,
`time`, `datetime`, `duration` or `list`), and `instructions`, a plain-language definition that we wrote once per
dataset and send to every system. A field can also carry hints: `unit`, `integer`, `decimals`, `items` (the item type
of a list), `dayfirst` and `number_locale`.

The system returns one value per field, or null when the document doesn't contain one. Many gold values are null on
purpose. Forms leave the PO number blank, most NDAs state no term, and half of the SQuAD 2.0 questions have no answer.
Telling these cases apart from real values is part of the task.

## 2. Input text

Every system reads byte-identical text for an item. Where the text comes from depends on the dataset:

| Text source | Datasets |
|---|---|
| Native text | Emails, TAT-QA, SQuAD 2.0, Kleister-NDA (the born-digital PDF text layer) |
| The dataset's official OCR | DocVQA, VRDU, Kleister-Charity (Textract), OmniAI (Azure Document Intelligence) |
| One shared Tesseract 5.5 pass, shipped in `datasets/<key>/ocr/` | FATURA, RealDoc, SROIE, CORD (`-l eng --oem 1 --psm 3`, retried with psm 6 on empty pages) |

Tesseract finds no text on 4 SROIE pages and 4 CORD pages. For those items, the runner records every field as null for
systems that read text only. It doesn't drop the item.

Results that also give the system the page image (`clairic_img`, which is Clairic with the image plus the same text) go
on a separate board, `extract-bench report --board image`. They never mix with the text board.

## 3. How each system is asked

Each system gets the same field definitions in its own request format. The adapters live in
`src/extract_bench/clients.py` and `open_extractors.py`, and [ADAPTERS.md](ADAPTERS.md) describes them.

| System | Request | Long documents |
|---|---|---|
| Sage 2 | `POST /v1/extract` with typed fields (bare questions on DocVQA and SQuAD 2.0), fast tier | Windows of at most 15k o200k tokens, split again on HTTP 413 |
| Clairic | Workers AI `@cf/cloudflare/clairic` with typed fields and `instructions`, default `null_threshold` | One 45k-token window |
| NuExtract3 | A JSON template with NuExtract types, the definitions passed as the `instructions` chat-template argument, thinking off | 32k context; the runner skips longer prompts |
| Schematron-3B | A draft-07 JSON Schema with the definitions as descriptions, the model card's prompt, the text wrapped in minimal HTML, schema-constrained decoding | 32k context |
| GLiNER2.5 | One entity label per field, with the definition as its description. A scalar field takes the most confident span; a list field takes every span | 1,800-character chunks that overlap by 300 characters |

The open-weight models decode greedily, at temperature 0. Nobody tuned any system's prompt on test items. Levanto
fitted Sage 2's abstention thresholds on the `fit` half (see §6).

## 4. Scoring one field

`score.score_item(item, preds)` gives every field one of four outcomes. The headline rule, rule (a), treats a null
prediction as an abstention:

| Gold | Prediction | Outcome | Score |
|---|---|---|---|
| null | null | correct abstain | 1 |
| null | a value | invented value | 0 |
| value | null | missed value | 0 |
| value | a value | answered | the metric's score, from 0 to 1 |

Rule (b) also counts an answer with confidence below 0.5 as an abstention. The scorer computes it for every system
that returns confidences, but the board doesn't use it.

The metric follows the original benchmark of each dataset. Nine sets use `exact_norm`, an exact match after normalising
for the field's type:

- The scorer NFKC-normalises text, folds dashes and quotes to ASCII, folds case, collapses whitespace and strips
  surrounding punctuation. Two texts match if they are equal as they stand, after dropping one leading article, or
  after squashing to letters and digits with the same digit runs (`'#1884'` equals `'1884'`, but `'2.03(b)'` doesn't
  equal `'203(b)'`).
- The scorer parses numbers and money from written forms such as `$1,234.56`, `1.234,56` and `(12.5)`, and compares
  money in cents, rounding half up.
- It parses dates, times and datetimes to ISO, taking `dayfirst` from the dataset card. For predictions only, it also
  reads contract phrasing such as "15th day of January, 2019".
- It reads durations as ISO 8601 (`P2Y`) or as words ("2 years") and compares them in seconds.
- List fields score the multiset F1 of their normalised items.

DocVQA uses `anls` with τ = 0.5. SQuAD 2.0 uses `token_f1` with the SQuAD v2 normalisation. TAT-QA uses `numeric_tol`,
which accepts a number within max(0.005, 1e-4 × |gold|) and scores span answers by token F1.

When a field has several acceptable gold values, such as FATURA's two buyer blocks or DocVQA's answer variants, the
best match counts. A dataset can mark some fields as not headline. On Kleister-NDA the headline fields are the
effective date, the jurisdiction and the term, and only headline fields count on the board. The scorer leaves a field
out, rather than counting it against the system, when the system can't express that field type or when the gold value
itself doesn't parse (`gold_unparseable`).

## 5. From fields to the board

1. The board uses the items in the version's split, `splits/v0.1.json`, and only their eval half.
2. Every system in a comparison answers the same fields. The board keeps only the documents that every system read
   without an error, and within each document only the fields that every system was scored on. A request that
   errored removes that document for all systems. Output that didn't parse (`unparsed: true`) stays in, and all its
   fields score 0.
3. Field accuracy per dataset is the micro-average over the kept fields.
4. The headline is the unweighted mean over datasets, leaving out the two receipt sets (§7). `board.json` also has the
   mean over all 12 sets.
5. "Not in the document" correct is the accuracy on fields whose gold is null, averaged over the datasets that have
   such fields.

## 6. Splits

`splits/v0.1.json` holds the items that meet three conditions:

1. They are outside each dataset's sealed 20% holdout (`common.is_holdout`).
2. They were scored in the Sage 2 read of 28 Oct 2026.
3. They fit in one Sage 2 request, meaning the 25 Oct 2026 reference read needed one call. This mostly removes long
   Kleister-Charity reports. One system defines this filter; KNOWN_ISSUES.md discusses it.

The `eval` half holds the items where int(sha1(item id), 16) is odd, and `fit` holds the rest. Levanto fitted Sage 2's
abstention thresholds on `fit`, so the headline numbers use `eval` only. `extract-bench report --split all` scores both
halves together.

## 7. Receipts

On SROIE and CORD, the shared Tesseract text misses much of the gold, so OCR caps every system that reads text only.
The dataset cards include the reachability checks. The board shows both sets but leaves them out of the headline mean.

## 8. Uncertainty

For each dataset, the board reports an item-paired bootstrap of the difference between a system and the reference.
It resamples documents with replacement, keeps each document's fields together, runs 2,000 resamples with
`random.Random(20)`, and takes the 2.5th and 97.5th percentiles. A "✱" means the interval excludes 0.

For the mean of datasets, a stratified bootstrap resamples items within each dataset and recomputes the mean, with the
same number of resamples and the same seed.

v0.1 doesn't correct for multiple comparisons, doesn't cluster beyond the document (FATURA documents share 10
templates, and Kleister-Charity reports share charities), and runs each system once. KNOWN_ISSUES.md lists these
gaps. The package already has the tools for a stricter analysis: `score.paired_bootstrap` (clustered by document),
`score.holm` and `common.CLUSTER_KEY`.

## 9. What v0.1 doesn't measure

v0.1 doesn't put cost or latency on the board. Each prediction record keeps its wall time and token usage, but we
didn't measure list prices under a common protocol, and Sage 2's extraction price isn't public. Image input appears
only on the separate image board. Documents longer than one Sage 2 request are outside v0.1 (§6).
