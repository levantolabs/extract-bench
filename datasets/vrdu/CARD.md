# vrdu: VRDU Registration (full mixed-template test, 300) and Ad-buy (60)

`build.py` built this set on 2026-10-23. The build is deterministic, and two rebuilds gave byte-identical `items.jsonl` and `img/`. It cost $0. No metered API was called.

| | |
|---|---|
| Items | 360 in total: 300 Registration documents (all mixed-template test documents) and 60 Ad-buy documents (seed 20, stratified by page count) |
| Use cases | Registration covers government and regulatory forms (US DOJ FARA filings: Amendment, Short-Form, Dissemination Report). Ad-buy covers TV political ad-buy orders, contracts and invoices, which are PO-like and carry line items. Ad-buy is the suite's only multi-page vision test, with up to 4 pages |
| Modality | Text plus image. `text` is the official OCR (`text_source: official_ocr`). `images` holds the PDF pages rendered as JPEG, at most 4 |
| Fields | 2,640 fields in total. Registration has 6 per document. Ad-buy has 9 scalars plus 5 line-item list fields per document. 591 have null gold (517 scalar plus 74 list) |
| Metric | `exact_norm` (harness, typed) for the paired head-to-head. The official VRDU micro-F1 (`tools/official_eval.py`) is the only number that may sit next to the published rows |
| LMDX-comparable | Registration only. It uses the same 300 documents, the same gold and the same official scorer as LMDX Table 2. The prompts differ (see "Comparability") |

## Source and revision
- The dataset is https://github.com/google-research-datasets/vrdu, commit `04a4df7ac5cc694e51c48115c03fe428f0e240cc` (2023-02-08, "dataset.jsonl.gz needs to be decompressed first"). The repo has had no pushes since then (`pushed_at` 2023-02-08T19:43:12Z).
  - We verified that `main/dataset.jsonl.gz` and `main/meta.json` for both corpora are byte-identical to the git blob SHAs of that commit:
    - registration `0e96947…`
    - ad-buy `224b1ea…`
  - They are hardlinked from `extract-bench/raw/kie-docs/vrdu/`.
  - We downloaded the 360 sampled PDFs from `raw.githubusercontent.com` at the same commit and checked each blob SHA against `raw/repo_tree_04a4df7.json`, the GitHub tree API listing.
- The 24 `*mixed_template*` split files (12 per corpus) are in `raw/*/few_shot-splits/`.
  - Within each corpus, all 12 files share one identical 300-document test list, with zero overlap with train or valid (`tools/check_splits.py`).
  - So "the mixed-template test set" is unambiguous.
- The official evaluator is https://github.com/google-research/google-research/tree/master/vrdu at commit `600df90d3b8875e62efa9d462faa9d87276fad74`. The raw copy is in `raw/official-eval/`, and `match_utils.py` and `benchmark_utils.py` are vendored verbatim in `tools/vrdu_official/vrdu/`.
  - A diff against the original 2023-01-30 release (`ff4f8b0`) shows that only the copyright year and `pyrefly` comments changed. So the matching logic is unchanged since LMDX's runs.
- We downloaded both papers as PDFs and read the tables below from them:
  - VRDU, https://arxiv.org/abs/2211.15421 (v3, 16 Sep 2023, KDD '23).
  - LMDX, https://arxiv.org/abs/2309.10952 (v2, 21 Jun 2024).

## Licence
- The dataset licence is UNKNOWN.
  - The repo has no LICENSE file. The GitHub API reports `license: null`, and `/LICENSE` at the pinned commit returns 404.
  - The README says the documents are public records from the US DOJ FARA site (https://www.justice.gov/nsd-fara) and FCC Public Files (https://publicfiles.fcc.gov/). I did not open those two pages, and the links come from the README.
  - Private filers wrote the documents, so copyright in them is UNKNOWN.
- The official evaluator code is Apache-2.0 (google-research LICENSE at `600df90d`, which I opened).
- Redistribution is not confirmed. See `LICENSE-DATA.md`.

## Size and sampling (seed 20)
| Corpus | All docs | Mixed test | In this set | How |
|---|---|---|---|---|
| Registration (FARA) | 1,915 | 300 | 300 | All of them. By template: Short-Form 138, Amendment 116, Dissemination Report 46. By page count: 1-page 20, 2-page 253, 3-page 27. All fit the 4-image cap |
| Ad-buy (DeepForm) | 641 | 300 | 60 | • We dropped and counted 29 test docs with more than 4 pages (the vision cap is 4 images), leaving 271 eligible.<br>• From those, `random.Random(20)` drew 30 one-page docs (of 81) and 30 two-to-four-page docs (of 190).<br>• `meta.stratum_weight` (1-page 0.598, 2-to-4-page 1.402) reweights to the test population of docs with 4 pages or fewer.<br>• Sample pages: 1-page 30, 2-page 11, 3-page 13, 4-page 6.<br>• 700 line items, p50 9 per doc, max 55 |

In our judgement, balanced Ad-buy strata work better here than proportional ones, so I chose them. With them, the 1-page versus multi-page vision comparison is not 18 versus 42. Reweight pooled Ad-buy numbers with `stratum_weight`. Ad-buy at n=60 is reference only, because it is a sample, it excludes the docs with more than 4 pages, and LMDX ran all 300.

The harness's default sealed 20% holdout applies. It uses a seed-20 hash of the id, and `meta.holdout` is not set. The bootstrap cluster is `meta.doc_id`, which is the filename.

## Field mapping
| Field | Type | Official matcher | Gold present / null |
|---|---|---|---|
| `file_date` | date (`dayfirst: false`) | DateMatch | 294 / 6 |
| `foreign_principle_name` (sic, official name) | text | NameMatch | 179 / 121 |
| `registrant_name` | text | NameMatch | 298 / 2 |
| `registration_num` | text | NumericalStringMatch | 299 / 1 |
| `signer_name` | text | NameMatch | 221 / 79 |
| `signer_title` | text | GeneralStringMatch | 78 / 222 |
| `advertiser` | text | GeneralStringMatch | 59 / 1 |
| `agency` | text | GeneralStringMatch | 28 / 32 |
| `contract_num` | text | NumericalStringMatch | 55 / 5 |
| `product` | text | GeneralStringMatch | 56 / 4 |
| `property` (TV station) | text | GeneralStringMatch | 55 / 5 |
| `tv_address` | text | AddressMatch (edit distance ≤ 3) | 51 / 9 |
| `flight_from`, `flight_to` | date | DateMatch | 45 / 15 each |
| `gross_amount` | money (`unit: USD`) | PriceMatch (±0.01) | 60 / 0 |
| `line_channel`, `line_program_desc` | list (items text) | GeneralStringMatch, inside the nested `line_item` | 43 / 17, 50 / 10 |
| `line_program_start_date`, `line_program_end_date` | list (items date) | DateMatch, nested | 53 / 7, 42 / 18 |
| `line_sub_amount` | list (items number) | PriceMatch, nested | 38 / 22 |

Each field also carries `official_match` (and `official_sub_entity` on list fields). The harness ignores these extra keys when it builds requests.

## How gold was built
- Text gold is the annotated text, with whitespace collapsed by the official `remove_redundant_whitespace`.
  - An entity can be annotated several times. Distinct mention texts become a list of acceptable values, and any one of them counts, as in the official matcher.
  - An absent entity becomes null. Scalar entities never repeat in these docs, which we checked.
- All 4,716 gold mentions in the 360 docs point at their text in the official OCR (0 mismatches, `meta.gold_offsets_check`).
- Span reachability is in `meta.gold_reachable.official_ocr`. The collapsed gold is a contiguous substring of the collapsed OCR text in almost every case. The exceptions are multi-segment entities that no single span can reproduce:

  | Field | Contiguous / total |
  |---|---|
  | registrant_name | 293/298 |
  | foreign_principle_name | 174/179 |
  | signer_name | 214/221 |
  | signer_title | 77/78 |
  | tv_address | 44/51 |
  | file_date (date; the 24 misses are multi-segment OCR-garbage golds such as `'1927 16th day of'`, a subset of the 43 in `meta.gold_issues`) | 270/294 |
  | All other fields (text and date) | 100% |

  (On 2026-10-23 the verifier added the missing `file_date` row to this table, from `meta.gold_reachable`.)
- For line items, the build sorts rows into document order, and each sub-entity becomes one parallel list.
  - A sub-entity that no row carries becomes null, not `[]`. That applies to 74 list fields, for example `line_sub_amount`, which is null in 22 docs.
  - `meta.official_line_items` keeps each row's official sub-entity tuple and order.
- **Declared date-gold repair.** The repair adds an acceptable alternative. The verbatim official gold always stays first, and the official scorer never sees the repair. The repair runs only where the harness's own `parse_date(dayfirst=False)` fails.
  - `token_expand` covers 12 flight dates and 203 line-item dates. Google's OCR merged ranges such as `02/27/20-03/02/20` into one token, and the gold holds character slices of it (`'02/27/'`, `'/02/20'`).
    - The verifier fixed one case on 2026-10-23. 4 line end-dates in `vrdu-adbuy-cda5811d-fe5aaa7a` were range tails such as `'20-01/02/20'` (from `01/01/20-01/02/20`). The harness wrongly parses these as 2002-01-20, so the repair had skipped them. `build.py` now also sends `RANGE_TAIL` (`^\d{1,4}-\d{1,2}/`) golds through `token_expand`, which gives `01/02/20`. The count went from 199 to 203. Only that one item changed. The images, the official round-trip and the harness ceilings are unchanged.
    - The repair expands the gold span to the enclosing OCR token and splits it on `-`. It keeps the piece that overlaps the gold span most, and accepts it only if it is a complete `MM/DD/YY[YY]` date.
    - For example, `'/02/20'` becomes `'03/02/20'`. We checked all 22 scalar repairs by eye.
    - Line-item fields with repairs get the gold `[verbatim list, repaired list]`, which is two acceptable lists.
  - `day_of_month`, `month_d_y` and `d_mon_y` cover 10 `file_date` golds. Written forms such as `'28th day of December 1976'`, `'June 11.2008'` and `'5. Jul 2013'` get an ISO alternative.
  - 43 `file_date` golds and 1 line date cannot be repaired. They are OCR garbage or partial spans, for example `'25 day of Octolico 68 168'`, `'(Date of signature'` and `'11'`. `meta.gold_issues` lists them, and they stay as the official gold.
    - So a typed (ISO) date arm tops out at 0.854 on `file_date` (`tools/harness_ceiling.out.json`). A verbatim-echo arm tops out at 1.0.
    - The verifier found more on 2026-10-23. 4 more `file_date` golds are OCR errors that the harness parses into impossible years, and they are not in `meta.gold_issues`. A typed arm that reads the real date gets 0 on them, so the realistic typed ceiling is at most 247/294 = 0.840. The 0.854 ceiling counts them as hits only because it echoes `parse_date(gold)`. These are the 4 cases:

      | Filename | Gold | Parsed by the harness |
      |---|---|---|
      | `19691101_Italian Government To…` | `'Subscribed and sworn to before me at October 19 691 31 day of'` | 0691-10-19 |
      | `19930401_KOTRA_Dissemination R…` | `'April 27, 193'` | 0193-04-27 |
      | `20150707_Korea Economic Instit…` | `'07/06/215'` | 0215-07-06 |
      | `20171010_MSLGROUP Americas, In…` | `'10121116'` | 1012-11-16 |

      The verifier found them by comparing each parsed `file_date` year with the filename's year. The test `|Δ| > 1` gave 9 hits. The other 5 parse to plausible years and were left alone. The items were not changed, because the harness does not read `gold_issues`.
- A null-gold audit (`tools/null_audit.py`) checked 30 of 517 null scalar slots, drawn with seed 20. The context is in `tools/null_audit.out.txt`. In our judgement, the verdicts are:

  | Verdict | Count | Fields |
  |---|---|---|
  | Clearly correct (absent from the OCR text) | about 23 | All 13 `signer_title`; 7 of 9 `foreign_principle_name`, where only boilerplate mentions the term; the agency, product and tv_address samples |
  | Probably a missed label | 2 | `foreign_principle_name`, where the text names a principal that was renamed or terminated |
  | Debatable | 5 | 4 `signer_name` where only a handwritten signature exists, so the name is absent from the text but may be legible in the image; 1 `flight_from` where only an invoice billing period is printed (the VRDU paper lists flight-vs-invoice-period confusion as a known annotator error) |

  The VRDU paper (§4.2.4) also reports missed last address lines as a common annotator error.

## Text and images
- The text is `dataset.jsonl` `ocr.text`, the OCR that the benchmark provides.
  - VRDU footnote 3 and LMDX footnote 1 say it is Google Cloud Vision OCR (I did not open those links).
  - The LMDX and VRDU baselines used this same OCR.
  - We did not re-OCR it, so OCR quality stays fixed. Real OCR noise shows in old typewritten FARA filings.
- The images come from `pdftoppm` in poppler 26.03.0, which renders pages 1 to min(4, n) as JPEG at quality 85.
  - DPI is at most 200, and the build lowers it only where needed to keep the long side at or under 2,200 px (the lowest DPI used is 69, for oversized pages). A typical page is 1,696 × 2,200 px.
  - There are 722 images, all at quality 85. No item needed the fallback ladder.
  - The largest image is 2,049,359 B and the largest per-item total is 4,026,021 B, so every item is under the 4 MiB-per-image and 8 MiB-total caps.
  - For each item, `meta.images` records DPI, pixel size, bytes and total.
  - PDF page count equals OCR page count on all 360 docs.
- Token counts (o200k, `tiktoken` 0.14.0, text only) are:

  | | p50 | p95 | max |
  |---|---|---|---|
  | Registration | 1,174 | 1,594 | 1,908 |
  | Ad-buy | 1,358 | 4,258 | 7,028 |
  | All | 1,174 | 2,489 | 7,028 |

  No item comes near the Sage 2 cap of 16,384 tokens (content plus field definitions), so no windowing is needed.
- The offline cost estimate comes from the harness `estimate.py`, at list prices and with its stated assumptions. Clairic image tokens are UNKNOWN. The estimate is about $12 for all 8 arms on 360 items:

  | Arm | Est. $ |
  |---|---|
  | clairic_text | 0.45 |
  | clairic_vision | 0.82 |
  | clairic_vision_ocr | 1.24 |
  | sage_precise | 3.87 |
  | sage_fast | 3.87 |
  | haiku_json | 0.11 |
  | gemini_vision_json | 1.17 |

## Metrics
1. **`exact_norm` (harness, primary for paired arms).**
   - It is type-aware. It parses dates with `parse_date(dayfirst=False)` and money with `parse_number`, and it scores lists with multiset `list_f1`.
   - Abstention follows rules a/b, under which a null gold makes an abstention correct.
   - Every arm gets the same items and the same text, so it is the fair head-to-head.
2. **Official VRDU micro-F1.** Run it with `tools/official_eval.py --run RUN --arm ARM [--rule a|b]`, which needs `--with editdistance`.
   - It reads the untouched official annotations from `raw/` and calls the vendored official `evaluate()`.
   - A null prediction means no extraction and is not counted. A value on an absent entity is a false positive.
   - The adapter is declared and does three things:
     - It uses the arm's verbatim text (`pred.text`, or `state[span]`) when it has one, otherwise `str(value)`.
     - It rewrites ISO dates as `YYYY/MM/DD`, because the official `DateMatch` has no ISO pattern. Only the format changes.
     - It rebuilds Ad-buy line items by index from the 5 parallel lists.
   - In the gold round-trip (`tools/official_eval_roundtrip.out.json`), Registration scores 1.000 micro-F1 under the official code, so the items' gold reproduces the official annotations exactly.
3. **Why both.** The official matchers are lenient in ways that differ from ours. GeneralStringMatch compares alphanumerics only. NameMatch ignores whitespace. AddressMatch allows an edit distance of up to 3. DateMatch accepts a day/month swap.

### Official line-item F1 is order-sensitive (verified), and this matters for Ad-buy
- `benchmark_utils.get_matching_result_per_doc` matches nested entities by the ordered tuple of sub-entity names.
- The gold uses 68 orders for 20 sub-entity sets. The gold order equals OCR reading order in 4,238 / 4,238 multi-field line items (`tools/line_order.py`).
- So a line item matches only if the extractor emits its sub-entities in reading order. In our judgement, this would favour the localizing LMDX models in LMDX Table 2 over non-localizing baselines such as GPT-4V (line-item F1 4.45).
- Ad-buy micro-F1 also includes line items, which outnumber scalars, so the published Ad-buy micro-F1 is mostly a line-item score.
- These are the ceilings for a perfect extractor on our 60 docs (official code unless noted):

  | Emission | micro-F1 | line-item F1 | scalar (`unrepeated_f1`) |
  |---|---|---|---|
  | Gold rows in gold order | 0.952 | 0.922 (official grouping still merges consecutive partial rows) | 1.000 |
  | Gold rows in fixed schema order | 0.569 | 0.290 | 1.000 |
  | Our parallel lists, schema order | 0.562 | 0.277 | 1.000 |
  | Our parallel lists, reading order (`--emit reading`, the default). Uses the arm's own list spans if returned, otherwise a gold-free OCR search | 0.751 | 0.591 | 1.000 |
  | Our parallel lists, `--line-order insensitive` (our fix, not official) | 0.916 | 0.862 | 1.000 |

- For Ad-buy, report these:
  - The headline is scalar `unrepeated_f1` (official) plus `exact_norm` on scalars.
  - Report line-item F1 separately, with the ceiling printed beside it.
  - Sage 2 has no list type in the 20 Oct spec, so line fields are `unscored` for it (counted in coverage-adjusted scores).

## Published baselines
All zero-shot rows come from LMDX v2, Table 2 (https://arxiv.org/abs/2309.10952), as micro-F1 from the official VRDU tool.
- Google (the method authors) ran them, so they are not independent.
- They use the full 300-doc mixed-template test, with |D| = 0 (zero-shot).
- The prompt (Figures 7 and 8) was raw benchmark OCR text or the page image, plus a schema of key names only. The instructions included "Use null or [] for missing entity types". There were no field descriptions.
- The models were GPT-4V (`gpt-4-1106-preview`), GPT-3.5 (`gpt-3.5-turbo-1106`) and Gemini Pro via Vertex (`gemini-pro-vision` for images).
- How multi-page docs were fed to the image baselines is UNKNOWN. The paper says "the document page as image input".

| System | Score | Setting | URL |
|---|---|---|---|
| Gemini Pro + OCR | 69.41 | Registration mixed, micro-F1, zero-shot, text | https://arxiv.org/abs/2309.10952 |
| GPT-4V + Image | 65.34 | Registration mixed, micro-F1, zero-shot, image | same |
| GPT-3.5 + OCR | 63.86 | Registration mixed, micro-F1, zero-shot, text | same |
| PaLM 2-S + OCR | 59.78 | Registration mixed, zero-shot, text | same |
| Gemini Pro + Image | 48.60 | Registration mixed, zero-shot, image | same |
| LLaVA-v1.5-13B + Image | 5.00 | Registration mixed, zero-shot, image | same |
| LMDX (Gemini Pro) | 75.15 | Registration mixed, zero-shot, LMDX prompt with layout coordinates (not raw OCR) | same |
| LMDX (PaLM 2-S) | 71.65 | Registration mixed, zero-shot, LMDX prompt | same |
| Gemini Pro + OCR | 34.46 (line-item F1 19.25) | Ad-buy mixed, all 300, zero-shot, text. Reference only (our n=60, ≤4 pages) | same |
| GPT-4V + Image | 31.95 (line-item 4.45) | Ad-buy mixed, zero-shot, image. Reference only | same |
| GPT-3.5 + OCR | 30.05 (line-item 7.65) | Ad-buy mixed, zero-shot, text. Reference only | same |
| LMDX (PaLM 2-S) | 39.74 (line-item 21.21) | Ad-buy mixed, zero-shot, LMDX prompt. Reference only | same |
| LMDX (Gemini Pro) | 38.02 (line-item 23.29) | Ad-buy mixed, zero-shot, LMDX prompt. Reference only | same |
| FormNet / LayoutLMv2 / LayoutLMv3 | 90.51 / 89.19 / 89.77 | Registration mixed, fine-tuned on 200 docs. Ceiling reference, not zero-shot | https://arxiv.org/abs/2211.15421 (Table 4) |
| LayoutLMv2 / LayoutLMv3 / FormNet | 46.54 / 45.16 / 43.23 | Ad-buy mixed, fine-tuned on 200 docs. Reference | same |
| GPT-4 + OCR | 43.7 | VRDU ad-buy, random train/test split, zero-shot, run by JPMorgan (DocLLM Table 5). Not comparable (different split and prompt) | https://arxiv.org/abs/2401.00908 |

## Comparability (read before quoting)
- Our arms' official Registration micro-F1 on these 300 docs is comparable with the Registration rows above. The docs, gold, OCR text and scorer are the same.
- Some things are not the same.
  - Our arms get typed fields with instructions (field descriptions), whereas LMDX gave key names only.
  - The models are three years apart.
  - The official score depends on the adapter (verbatim text vs typed value). Report the arm's verbatim-text variant.
  - Call it "same items, same scorer, different prompt", never "beats GPT-4V" without that caveat.
- Ad-buy is reference only. It is a sample of 60, limited to docs with 4 pages or fewer, and its line items are order-sensitive (see above).
- Sage 2 (20 Oct spec) reads only the official OCR text. It has no image input and no list type.
- Clairic vision gets every page of every item. Registration has at most 3 pages, and Ad-buy has at most 4 by construction.

## Contamination risk
- In our judgement, the risk is medium-high.
  - OCR text and gold have been public on GitHub since 2023-02-08.
  - The source PDFs are public on DOJ and FCC sites.
  - The papers print example documents.
- VRDU is not among the tasksource extraction-corpus sources (a check by source name under suite design decision L1). Re-run that check on the day the new Sage is benchmarked.

## Files
| Path | What |
|---|---|
| `items.jsonl` | 360 items in the common format. sha1 `b6d8e0eb…` after the 2026-10-23 verifier fix (was `ff8d4eb6…`) |
| `build.py` | Rebuilds `items.jsonl` and `img/` from `raw/`. `--fetch` downloads missing PDFs from the pinned commit and checks their blob SHAs. Run it with `/opt/homebrew/bin/uv run --no-project --with tiktoken --with pillow python -I extract-bench/datasets/vrdu/build.py` |
| `build_info.json` | Sampling counts and stratum weights |
| `img/` | 722 JPEGs (352 MiB) |
| `raw/` | 131 MB on disk, of which 86 MB is hardlinked jsonl.gz:<br>• `{registration-form,ad-buy-form}/main/{dataset.jsonl.gz, meta.json, pdfs/}` (46 MB of PDFs)<br>• the split files<br>• `official-eval/` (pristine)<br>• `repo_tree_04a4df7.json` |
| `tools/official_eval.py` | Official VRDU scorer adapter (`--gold-roundtrip`, `--run/--arm`, `--rule`, `--emit`, `--line-order`) |
| `tools/selfcheck.py` (writes `selfcheck.out.json`) | Structure, images, tokens, gold parse, nulls, reachability (0 errors) |
| `tools/harness_ceiling.py` (writes `.out.json`) | Harness-side ceilings: verbatim 1.0 on every field; typed `file_date` 0.854 |
| `tools/null_audit.py` (writes `.out.txt`); `tools/date_gold_audit.py`; `tools/line_order.py`; `tools/check_splits.py`; `tools/stats_test.py` | Audits behind the claims above |

## Open items
- Harness note, not edited here: `score._is_null` does not treat `[]` as an abstention, so a list arm returning `[]` for a null-gold line field scores as "hallucinated". The suggestion is to treat `[]` as null for `type: list`. *(Resolved on 2026-10-23. `score._is_null` treats `[]` as an abstention on list fields.)*
- Clairic image-token billing is UNKNOWN. The C1 probe gates the vision arms.
- Whether the new Sage version adds vision or lists is UNKNOWN. If it does, re-cut the Sage arms.
- The licence is UNKNOWN.
