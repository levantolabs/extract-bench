# Adapters, arms and official scorers

An arm is one system with one input configuration. `extract-bench run read --arms A,B` reads every item with each arm
and writes one JSON file per arm, dataset and item to `runs/RUN/preds/ARM/DATASET/ID.json`. You can stop and restart a
run at any point, because the runner skips files that already hold a good answer.

## Arm names

An arm name has the form `BASE[@variant][#view]`.

| Base | System | Input |
|---|---|---|
| `sage_fast`, `sage_precise` | Sage 2, `POST {SAGE2_BASE_URL}/v1/extract` | Text and scalar fields. Skips the sealed 20% holdout unless you pass `--final` |
| `clairic_text` | Cloudflare Clairic (`@cf/cloudflare/clairic`) | Text, in one window of 45k o200k tokens |
| `clairic_text_w15k` | Clairic | Text in 15k-token windows, the same windowing Sage 2 uses |
| `clairic_text_whole` | Clairic | Text in one 58.5k-token window |
| `clairic_vision` | Clairic | The page image only |
| `clairic_vision_ocr` | Clairic | The page image plus the same text the text arms read |
| `haiku_json`, `gemini_vision_json`, `or:<model>:text\|image\|image+text` | Any OpenRouter model | A strict JSON schema built from the fields |
| `nuextract3`, `schematron3b`, `gliner25` | Open-weight models | Text, run through `extract-bench open` (export, then GPU inference, then import) |

A `@variant` reads an alternative text instead of the default one. Items that don't have that text count as not
applicable. The variants are `@tesseract_layer`, `@textract_layer`, `@djvu_layer` and `@best_layer` for Kleister;
`@gold_markdown`, `@aws-textract`, `@gemini-2.0-flash-001` and `@azure-document-intelligence` for OmniAI;
`@oracle_text` for CORD, which gives the gold words and is an oracle rather than an OCR ceiling; and `@pdf_text_layer`
for the 60 of 200 RealDoc items that have one. This repo doesn't ship the Kleister alternative layers.
`datasets/kleister_*/build.py` rebuilds them.

A `#comparable` view uses the source benchmark's own format, and only the official scorers read it. Never put these
results next to typed results. On SROIE it uses the four keys as verbatim text, on CORD it uses CC-OCR's image and
prompt, and on Emails it sends `prompt2` and its schema verbatim.

## Writing a new adapter

An adapter turns an item and its text into a request, and turns the response into
`{field_name: {"value": ..., "conf": float | None}}`. Give number and money values as JSON numbers, dates as ISO strings
and list fields as lists. Use `None` when the document doesn't contain the field. If the system can't express a field,
set `"unsupported": "<reason>"` and the scorer leaves that field out.

1. Add a request builder and a response parser to `clients.py`, next to `clairic_body` and `parse_clairic`.
2. Add a mock response (`mock_*`), so the arm runs offline.
3. Make live calls through `_post(...)` and record each one with `_ledger(...)`, so the spend cap counts it.
4. Add the base name to `ARMS` in `run.py`, with its `system`, its `input` and any options.
5. Add a mock-mode test in `tests/`.

You can also skip the adapter and submit predictions that you produced elsewhere, as [SUBMITTING.md](../SUBMITTING.md)
describes.

## Per-dataset rules

The runner applies these rules, which come from each dataset's CARD:

- Sage 2 always uses bare questions on DocVQA and SQuAD 2.0. On other sets it follows `EXB_SAGE_SHAPE`, which
  defaults to `typed`.
- Sage 2 doesn't score TAT-QA computed answers (`na_arms`). An item that has no field a system can express doesn't
  apply to that system.
- At load time the runner adds field hints in memory without overwriting any key a field already has. It takes list
  item types from `meta.field_meta` or `meta.list_item_types`, and gives CORD money fields the dot-thousands locale and
  the IDR unit.
- The scorer parses contract phrasing such as "15th day of January, 2019" in predictions only. It treats a year
  outside 1800 to 2199 as unparseable, and reads ambiguous RealDoc dates month-first.
- `score.paired_bootstrap` clusters items by `common.CLUSTER_KEY`. That is `charity_cluster` for Kleister-Charity,
  `cluster_id` for CORD and `template` for FATURA. Other sets use `meta.doc_id`, or the item itself.

## Official and comparable scorers

Run `extract-bench run export --run RUN` first. Then, from the repo root:

| Dataset | Command |
|---|---|
| Emails | `python -I datasets/emails/comparable/score_emails.py runs/RUN/export/ARM/emails.items.jsonl runs/RUN/export/ARM/emails.score_emails.jsonl` |
| SQuAD 2.0 | `python -I datasets/squad2/comparable/squad_v2_eval.py datasets/squad2/items.jsonl runs/RUN/export/ARM/squad2.squad_preds.json` |
| Kleister-NDA, Kleister-Charity | `cd datasets/KEY && python -I comparable/kleister_geval.py --preds-dir ../../runs/RUN/preds/ARM/KEY [--exclude-holdout]` |
| TAT-QA | `uv run --with numpy --with pandas --with scipy python -I datasets/tatqa/comparable/run_official.py --run RUN --arm ARM` |
| VRDU | `uv run --with editdistance python -I datasets/vrdu/tools/official_eval.py --run RUN --arm ARMDIR` |
| RealDoc | `realdoc_score.summarize(items, preds)` from `datasets/realdoc/scripts/` (needs `rapidfuzz`) |
| SROIE, CORD | Feed `runs/RUN/export/ARM/KEY.ccocr.json` to CC-OCR's `KieEvaluator` with `comparable/label.json` (needs `zss` and `nltk`) |

We wrote the Emails, SQuAD and Kleister scorers ourselves, using only the standard library. The TAT-QA, VRDU, RealDoc
and CC-OCR scorers wrap code downloaded from upstream, so review that code before you run it.
