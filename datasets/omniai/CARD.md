# omniai: OmniAI OCR Benchmark, use-case-weighted [vendor-built, secondary]

We built this set on 2026-10-23 for extract-bench. It adapts an existing set, and nothing here is built from scratch. We called no metered API.

In our judgement, this is usable as a secondary set that maps page images to typed fields, and it covers nine of the suite's use cases. It has one property no other set has. The same page comes with gold text, three commercial OCR texts and nine published vendor-run pipelines. It also has these weaknesses:
- It is vendor-built and partly synthetic.
- It has been public since Feb 2025.
- In this sample, 72% of the gold values sit in list fields, which Sage 2 (20 Oct spec) cannot score.

## Source and version
| What | Value |
|---|---|
| Dataset | https://huggingface.co/datasets/getomni-ai/ocr-benchmark, revision `4ed0d95271ca00107726230f7a0944ed9e90d897` (lastModified 2025-02-21). The files are `test/metadata.jsonl` (LFS sha256 `dad8fc25…9814b`, verified) and `test/images/<id>.(png\|jpg\|jpeg)` |
| Per-provider OCR and predictions | https://huggingface.co/datasets/getomni-ai/ocr-leaderboard, revision `5bedb06b13b42743cbb4e2f088ec24da6cc10ca4`. `results.csv` has 8,932 rows and 132,076,489 bytes, with LFS sha256 `dbce98aa…43dbc` (verified). The repo also holds `leaderboard_overview.csv` |
| Code and scorer | https://github.com/getomni-ai/benchmark. The head is `cf84a584745297e97f53921e1b8896522fd3c75b` (2025-10-21). The Feb-2025 release commit is `3058c1f9d6253b311446212738a905d1df313e58` (2025-02-19). We saved `json.ts` and the prompts at both commits in `raw/github/` |
| Results page | https://getomni.ai/ocr-benchmark is dead. It returns a 308 redirect to `monumint.com/ocr-benchmark`, which returns 404 (checked 2026-10-23). The archived copy is https://web.archive.org/web/20250223212005/https://getomni.ai/ocr-benchmark (saved as `raw/wayback/`) |
| Split | `test` is the only split, with 1,000 single-page images |

## Licence
- The licence is MIT. The HF card front matter of both repos says `license: mit` (saved as `raw/README_*.md`), and the GitHub `LICENSE` says MIT (saved as `raw/github/LICENSE`).
- In our judgement, the licence has a caveat. The archived results page says "The data used in the benchmark is a mixture of annotated and synthetic data", and the annotated part includes third-party images that OmniAI's MIT grant cannot relicense:
  - PHOTO_RECEIPT pages are SROIE receipts (see the risks section).
  - FORM and SCANNED_FORM pages look like tobacco-litigation documents.
  - One DRIVER_LICENSE is a 1965 California licence.
- This repo does not redistribute the page images; see `LICENSE-DATA.md`.

## What the set is
- Each page comes with an image, gold markdown, a JSON Schema, gold JSON and metadata. The metadata fields are `format`, `documentQuality` (CLEAN, HIGH_QUALITY, LOW_QUALITY or PHOTO), `fontFamily` and `rotation`.
- The full set has 1,000 docs in 37 formats. Of these, 491 fall in the 13 use-case formats used here.
- `results.csv` holds the vendor's Feb-2025 run of 9 pipelines. In these, 8 OCR providers each feed azure-gpt-4o extraction, and one more pipeline feeds the gold markdown to gpt-4o. Each row has the OCR markdown, the predicted JSON and a per-item `json_accuracy`.
- `file_url` in results.csv is an S3 UUID, not the benchmark id, so we join rows on the canonical (gold JSON, schema) pair. Of the 8,932 rows, all 8,932 join. The 1,000/1,000 ids match one-to-one, and none are ambiguous.

## Sampling 240 items with seed 20
The sampling follows suite design decision C4.

1. A doc is eligible only if it passes all of these tests:
   - Its format is one of the 13 use-case formats.
   - It has Azure DI text in results.csv. Only doc 38 (ACCOUNT_STATEMENT) lacks it.
   - It has at least one non-boolean scalar field after mapping. The 25 multi-check BANK_CHECK pages fail this test (see the BANK_CHECK adaptation below).
2. We take every eligible doc in six formats: SHIPPING_INVOICE 52, DELIVERY_NOTE 51, PHOTO_RECEIPT 10, DRIVER_LICENSE 4, FORM 7, and SCANNED_FORM 6, which makes 130 docs.
3. We share the remaining 110 evenly over the other 7 formats, and 110 is 7 times 15 plus 5. So the first five formats in spec order get 16:
   - BANK_CHECK 16, ACCOUNT_STATEMENT 16, CREDIT_CARD_STATEMENT 16, FORM_1040 16, and PATIENT_INTAKE 16.
   - COMMERCIAL_LEASE_AGREEMENT 15, PETITION_FORM 15.
4. Within each even-share format, PHOTO-quality and rotated docs go in first (21 docs). We draw the rest with `random.Random(20).sample` from the id-sorted pool (89 docs).

| Format | Items | PHOTO | Rotated | Scalar fields (null) | List fields (null) | Booleans excluded |
|---|---:|---:|---:|---:|---:|---:|
| SHIPPING_INVOICE | 52 | 2 | 3 | 552 (12) | 136 | 0 |
| DELIVERY_NOTE | 51 | 1 | 3 | 765 | 639 | 0 |
| PHOTO_RECEIPT | 10 | 10 | 0 | 89 (7) | 43 | 0 |
| DRIVER_LICENSE | 4 | 4 | 0 | 64 (5) | 0 | 0 |
| FORM | 7 | 7 | 0 | 62 | 22 | 1 |
| SCANNED_FORM | 6 | 6 | 0 | 113 | 6 | 1 |
| BANK_CHECK | 16 | 1 | 1 | 240 (6) | 0 | 0 |
| ACCOUNT_STATEMENT | 16 | 0 | 3 | 106 | 98 | 0 |
| CREDIT_CARD_STATEMENT | 16 | 0 | 3 | 57 | 71 | 0 |
| FORM_1040 | 16 | 1 | 3 | 444 (28) | 64 (12) | 265 |
| PATIENT_INTAKE | 16 | 3 | 2 | 150 (22) | 48 (14) | 190 |
| COMMERCIAL_LEASE_AGREEMENT | 15 | 1 | 3 | 240 | 0 | 0 |
| PETITION_FORM | 15 | 0 | 0 | 150 | 60 | 0 |
| **Total** | **240** | **36** | **21** | **3,032 (80)** | **1,187 (26)** | **457** |

- The gold has 2,952 non-null scalar values and 7,712 values inside list fields, so 72.3% of gold values are in lists.
- The Sage 2 head-to-head covers scalars only, so it covers a minority of what the page asks for. We report that as a limit.

## Text (how we produced `text`)
- `text` is the Azure Document Intelligence markdown, byte-identical to `results.csv` (`ocr_model=azure-document-intelligence`), with `text_source: "official_ocr"`.
  - It comes from the vendor's Feb-2025 run.
  - The archived page says traditional OCR used default configs, "with the exception of Azure which supports a native Markdown output".
- We save alternative texts per item and reference them from `meta.ocr_alternatives`:
  - `ocr/gold_markdown/<id>.md` is the dataset's gold transcription. It is the labelled ceiling arm, never the headline.
  - `ocr/aws-textract/<id>.md` is missing for `omniai-0348`.
  - `ocr/gemini-2.0-flash-001/<id>.md` is missing for `omniai-0529`.
  - `ocr/azure-document-intelligence/<id>.md` is identical to `text`.
- Images are at `img/<id>.(png|jpg)`, one per item, with `page_count` 1. We keep rotated pages rotated, because rotation is part of the test.
  - 239 images are verbatim copies, and we check their sha256 against the HF LFS oid.
  - 131 sampled files named `.png` hold JPEG bytes, so we name output files by content (151 JPEG, 89 PNG).
  - One file is over 4 MiB. We re-encoded `407.png` (4,293,618 bytes) to JPEG q90 (624,737 bytes, same size in pixels).
- The o200k token counts per item are:

  | Text | p50 | p95 | max |
  |---|---:|---:|---:|
  | Azure DI `text` | 1,093 | 3,702 | 6,232 |
  | Gold markdown | 764 | 2,413 | 3,908 |
  | Field definitions (JSON with name, type and instructions) | 691 | 1,262 | 1,592 |

  - No item exceeds Sage's 16,384 cap. The largest text plus the largest field definitions come to 7,824.
  - The field count per item is p50 15, p95 30, and max 37. Clairic re-bills the state on each pass.

## Field mapping
- The mapping starts from the official JSON Schema and the gold JSON.
- We build each field name by joining the path with `__` and sanitising it to `^[A-Za-z_][A-Za-z0-9_]{0,63}$`. A name over 64 characters becomes a 55-character prefix plus an 8-hex sha1.
- For each field, the per-item `meta.name_map` and `meta.field_meta` hold the original path, schema type, Sage type, stratum, adaptation and date rule.

| Schema leaf | Field `type` | Gold | n (sample) |
|---|---|---|---:|
| string, plain | `text` | verbatim string | 2,078 |
| enum (`enum`, `type:"enum"`, or the schema typo `emum` on PATIENT_INTAKE maritalStatus) | `text`, with the options listed in the instructions | `[label, label with "_"→" "]` (e.g. `WELLS_FARGO`/`WELLS FARGO`) | 90 |
| string with `format: date`, or a name or description that says date, where every non-null gold for that path across the 13 formats also parses as a full calendar date | `date` | `[verbatim, ISO]`, or `[verbatim, ISO-MDY, ISO-DMY]` when day and month are ambiguous (both 12 or less) | 278 (36 ambiguous) |
| string whose name says `time`, where every gold value is a clock time | `time` | `[verbatim, HH:MM:SS, HH:MM]` (we read one gold, `1223`, as 12:23) | 10 |
| number | `number` | number | 576 |
| boolean | excluded, and kept in `meta.booleans_excluded.gold` for the decision-model side test (suite design decision C17) | n/a | 457 |
| array of scalars | `list` | values in order, with nulls dropped | |
| array of objects | one `list` per column (`items__price`, …), with nested scalar arrays flattened across entries | column values in order | 1,187 lists total |
| key null or absent in the gold JSON | same type | `null` | 80 scalar and 26 list |

We declared these rules before scoring:
- 2-digit years pivot at 30. For example, `18` becomes 2018, and `95` becomes 1995.
- Ambiguous numeric dates take month-first (US docs) as the primary reading and also accept day-first.
  - The strict variant drops the DMY alternative. The 36 affected fields carry `field_meta.date_rule = "mdy_ambiguous"`.
  - Several PHOTO_RECEIPTs are Malaysian, which means day-first.
- These date-like strings are not full dates, so they stay `text`:
  - the 1040 `taxDateStart` ("1/01");
  - the credit-card `transactions[].date` ("01/14");
  - the FORM `submission_date`.

We also declared two adaptations:
- **BANK_CHECK single-record flatten.** The schema wraps every check in a `checks[]` array. On the 27 pages whose gold has exactly one check, we turn that check's leaves into scalar fields (`check__checkInfo__payee`, …). The 25 multi-check pages are ineligible.
  - In our judgement, checks would contribute nothing to the scalar head-to-head without this.
  - The json.ts rebuild maps the fields back to `checks[0]`.
- **Lease `years` and `months`.** These stay `number`, tagged with stratum `duration_component` (30 fields).
  - The gold is one component of "X years and Y months", not a total duration, so the `duration` type would be wrong.
  - This set does not test the typed `duration` field.

The type map below follows suite design decision C2. It is a hint for the harness, stored in `field_meta.sage_type`.

| Field type | Clairic | Sage 2 (20 Oct spec) |
|---|---|---|
| text or enum | `text` | `text` |
| date or time | `date` / `time` | `date` / `time` |
| number | `number` | `money` 376 / `integer` 170 / `decimal` 30 |
| list | `list` (`items`: text 712, number 458, date 17) | not supported, so excluded from the head-to-head |

A critic found and fixed a bug on 2026-10-23. The item type lives in `meta.field_meta[name].item_type`, not on the field, so the harness had asked Clairic and the JSON-schema arms for text lists. It had also scored number lists as text. A correct `"6.20"` failed against gold `6.2`, and floats scored 1.0 on only 167/444 number lists. Now `harness/common.apply_hints` copies `item_type` into the field's `items` in memory, and floats score 1.0 on 444/444 (`../../scripts/critic_selfcheck_2026-10-23.out.json`).

- We choose `sage_type` for numbers like this:
  - `money` when the key tokens or the description contain a currency word (amount, price, total, balance, rent, deposit, tax, income, …) and no count word (items, quantity, number, years, …);
  - otherwise `integer`, when every gold value for the path is an integer;
  - otherwise `decimal`.
- This is a heuristic. Its word lists are in `build.py` (`MONEY_TOK`, `COUNT_TOK`).

The span ceiling matters because Clairic `text` answers are spans of the state.
- There are 2,091 text and enum scalar golds. After lower-casing, collapsing whitespace and stripping markdown `*|#`, the verbatim value appears in the Azure DI text for 1,994 (95.4%) and in the gold markdown for 2,044 (97.8%). The per-field flags are `field_meta.span_in_primary_ocr` and `span_in_gold_markdown`.
- Enum labels such as `SINGLE_FACE` never appear verbatim, so they rely on the spaced variant and the normaliser.

## Metric
| Score | Fields | Use |
|---|---|---|
| `exact_norm` (item `metric`) | Scalar fields. The gold is a value or a list of acceptable values, and a null gold means absent | The primary typed-field score, used for the Sage 2 vs Clairic head-to-head (3,032 fields). The harness owns the normaliser |
| `list_f1` (`field_meta.metric`) | List fields | Only for the Clairic, Haiku and VLM arms |
| OmniAI `json.ts`, Feb-2025 version (`raw/github/feb2025_src_evaluation_json.ts`, commit 3058c1f, before `ignoreCases` existed) | The full gold JSON in `meta.true_json` | This is the only number that may sit next to the published rows. Rebuild the predicted JSON from the field answers via `name_map`, zipping list columns back into records. Unanswered booleans count as deletions unless the decision-model side test fills them. A variant that strips booleans from both sides is "our adaptation", and it never goes in the same table |

- We keep both scores because the suite measures exact_norm per typed field, while the vendor's published numbers measure json.ts.
- json.ts counts an array element with one wrong value as two errors, as the archived page states. So the two scores are not interchangeable.
- We infer that the vendor scored with the Feb-2025 commit of json.ts. The results were uploaded on 2025-02-21 and the release commit is dated 2025-02-19. `json.ts` changed on 2025-04-02. Beyond that inference, the version is UNKNOWN.

The set also gives free reference rows that need no API calls:
- For every item, `published_predictions.jsonl` holds the vendor-run `predicted_json` and `json_accuracy` of all 9 pipelines.
- The harness should re-score these with its official exact_norm. That gives a $0 GPT-4o row on exactly these fields.
- To check the mapping, `scripts/sanity_published_scalar.py` re-scored them with a simple normaliser:
  - gpt-4o reading gold md: 0.997 on 3,032 scalar fields;
  - gpt-4o reading Azure DI: 0.934;
  - OmniAI: 0.950;
  - Textract: 0.907 (n=3,022);
  - Claude 3.5 Sonnet: 0.702.
- These are sanity numbers, not official ones.

## Published baselines
These rows give JSON accuracy with a GPT-4o extractor on all 1,000 docs, from Feb 2025. The vendor reports them itself, and OmniAI sells the top row.

| System (OCR, then extractor) | Score | Setting | URL |
|---|---|---|---|
| OmniAI, then gpt-4o | 0.92 (page: 91.7%) | full 1,000, json.ts | [leaderboard_overview.csv](https://huggingface.co/datasets/getomni-ai/ocr-leaderboard/blob/5bedb06b13b42743cbb4e2f088ec24da6cc10ca4/leaderboard_overview.csv); [archived page](https://web.archive.org/web/20250223212005/https://getomni.ai/ocr-benchmark) |
| Gemini-2.0-Flash, then gpt-4o | 0.86 (86.1%) | same | same |
| Azure DI, then gpt-4o (pairs with our primary text) | 0.85 (85.1%) | same | same |
| GPT-4o, then gpt-4o | 0.75 (75.5%) | same | same |
| AWS Textract, then gpt-4o | 0.74 (74.3%) | same | same |
| Claude 3.5 Sonnet, then gpt-4o | 0.69 (69.3%) | same | same |
| Google Document AI, then gpt-4o | 0.68 (67.8%) | same | same |
| GPT-4o-mini, then gpt-4o | 64.8% (on the page only, not in the CSV) | same | archived page |
| Unstructured, then gpt-4o | 0.51 (50.8%) | same | CSV; archived page |
| Gold markdown, then gpt-4o | "typically 99% (+/-1%)" on the page; our mean of the 1,000 per-item rows is 0.9951 | same | archived page; results.csv |

On this 240-item sample, the means of the vendor's per-item `json_accuracy` (results.csv) are below. `build.py` computes them and stores them in `build_stats.json`.

| Pipeline | Mean on sample |
|---|---|
| gold md | 0.9949 |
| OmniAI | 0.9257 |
| Azure DI | 0.8929 |
| Gemini | 0.8881 (n=239) |
| Textract | 0.8476 (n=239) |
| Google DocAI | 0.7545 |
| GPT-4o | 0.7279 |
| Claude 3.5 Sonnet | 0.6595 (n=238) |
| Unstructured | 0.5569 |

- The verifier found a denominator caveat on 2026-10-23. Each published row is the mean over the docs that pipeline returned, not over all 1,000 docs.
  - The results.csv row counts are ground-truth 1,000, Azure DI 999, Google DocAI 999, OmniAI 997, Unstructured 993, Textract 988, Gemini 987, Claude 3.5 Sonnet 987, and GPT-4o 982.
  - The recomputed means match the CSV and the page to 2 decimal places: Azure 0.8501, Gemini 0.8601, OmniAI 0.9174, Textract 0.7432, GPT-4o 0.7548, Claude 0.6934, DocAI 0.678 and Unstructured 0.508.
  - So a doc that fails is dropped, not scored 0. The harness must state its own failure rule, and it must report "failures dropped" whenever its results sit next to these rows.
- The sample is easier for traditional OCR than the full set, because it excludes charts and tables. Textract scores 0.10 higher on it (0.8476 against its full-set per-item mean of 0.7432).
- Any paired comparison must use the sample means, not the full-set rows.

## Contamination risk: high, in our judgement
- The images, gold markdown, schema and gold JSON have been public on HF since 2025-02-21. The predictions of 9 systems are public too.
- The archived page displayed 50 samples.
- The training data of Clairic (Oct 2026) and Sage 2 is UNKNOWN.
- PHOTO_RECEIPT draws on SROIE, which has been public since 2019.
- The set is not among the 127 tasksource extraction sources (checked under suite design decision L1). Re-check this when the new Sage ships.

## Risks and known issues
1. **Vendor-built.** OmniAI built the set, ran the baselines and tops the table. By the vendor's own statement, the set mixes annotated and synthetic data.
   - The S3 bucket is `synthetic-pdfs`, and the names look Faker-generated ("Lela_McLaughlin@yahoo.com").
   - Per-doc provenance (annotated or synthetic) is UNKNOWN, because the HF metadata has no such tag. The archived page says its data explorer shows one, but the snapshot renders only the first sample.
   - The verifier added a provenance proxy on 2026-10-23. `fontFamily` is missing on 124/1,000 docs. In this sample it is missing on exactly the four formats that look like real-world documents: PHOTO_RECEIPT 10, DRIVER_LICENSE 4, FORM 7, and SCANNED_FORM 6, for 27 items. It is present on all 213 other items.
     - The one doc the archived explorer tags `annotated` is doc 26 (SCANNED_FORM, in the sample as `omniai-0026`), and it has no `fontFamily`.
     - In our judgement, this means about 27/240 items are probably annotated real documents and about 213/240 are generated (synthetic) pages. The vendor has not confirmed this. Report the two groups separately.
   - Gold quality has not been audited. The vendor reports about 99% for gpt-4o reading gold md, and our sanity check gets 0.997.
2. **Cross-dataset duplicates.** `omniai-0009` and `omniai-0019` are the same receipts as SROIE test `X51006329183` and `X51005746203`. The company and date are equal. The totals are equal in value (27 vs "27.00" and 408.45 vs "RM408.45"), although the strict script's string test marks the second one `total: false` (`scripts/overlap_sroie_strict.out.json`). Both keys are present in `datasets/sroie/items.jsonl`. We flag both items in `meta.cross_dataset_duplicate`, and pooled suite stats must count each receipt once.
   - Three more PHOTO_RECEIPT store names (AEON PERMAS JAYA, BECON STATIONER, NASI KANDAR PELITA) occur in SROIE test with a different date/total, so they are probably other SROIE receipts.
   - A weak snippet probe found no FUNSD-test overlap.
3. **Mostly lists.** Lists hold 72% of gold values, and Sage 2 has no list type, so its head-to-head covers only the scalar minority.
4. **Small strata.** DRIVER_LICENSE has only 4 items, so ID and KYC remain a suite gap. There are 10 time fields, and no `duration` or `datetime` fields.
5. Of the 3,032 scalar golds, 80 (2.6%) are null, so this is not an abstention set and supports no abstention claim.
6. **Date leniency.** 36 date fields are day/month ambiguous and accept both readings. Report a strict variant.
7. The results page link is dead, so the archived copy is the only page-level source.

## Files
| Path | What |
|---|---|
| `items.jsonl` | sha256 `cd7c4e2e05337399d878a43f4b9f70982faf3e2d4fb848a78031e6485055d279`. The build is deterministic, which we checked by rebuilding |
| `build.py` | Re-creates everything from `raw/`. With `--offline` it fails rather than fetching. It fetches the sampled images from the pinned revision and checks them against the LFS sha256 |
| `raw/` | 229 MB, holding `metadata.jsonl`, `results.csv`, `hf_tree_images.json`, `images/` (240 sampled), the READMEs, `leaderboard_overview.csv`, `github/` and `wayback/` |
| `img/`, `ocr/` | 91 MB and 7 MB |
| `published_predictions.jsonl` | The vendor's per-item predictions and scores for the 9 pipelines |
| `build_stats.json` | Counts used in this card |
| `scripts/` | Exploration, self-check (`selfcheck.out.json` reports 0 errors), sanity re-score, overlap checks and card stats. Run them with `uv run --no-project --with pillow --with tiktoken python -I …` |
