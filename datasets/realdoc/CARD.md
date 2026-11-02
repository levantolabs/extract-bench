# realdoc: RealDoc-Bench QA (Extend) [vendor-built, secondary]

We built this set on 2026-10-23 at $0 and called no metered API. The only network use was free downloads from Hugging Face, GitHub, arXiv and extend.ai. The suite plan is in the suite design notes (§2.6, §8 row 7, and suite design decisions L9, L13, C2, C9, C16, C17).

- We sampled 200 questions from the official 1,356-question QA bank, stratified by domain (seed 20). They cover 180 single-page scanned or filled forms in mortgage (74), supply chain (52), finance/insurance (51) and medical (23).
- There are 573 scored fields: 424 text (117 of them enums), 134 number and 15 date. Of these, 39 have a null gold (6.8%), which means the value is blank or absent on the page. We exclude 16 boolean keys and route them to the decision side test, so 573/589 gold keys (97.3%) are scored.
- We render every page at 150 dpi, the same DPI as RealDoc's own VLM parser. The 180 page files are 178 PNG and 2 JPEG, at most 3.7 MB each, with 1 per item. `text` is null on every item, because there is no official OCR and 127/180 docs have no text layer. Sage 2 and Clairic-text will read the harness OCR text instead.
- The primary score is RealDoc's own `deep_equal`, with `fuzzy_equal` on plain-string keys. It runs a verbatim copy of the upstream code (`scripts/realdoc_official.py`). The item-level `metric: exact_norm` is only a harness fallback, and we never print it next to a published row.
- The published rows come from the vendor, and Extend ranks itself #1. They vary the parser while holding a Gemini 3 Flash extractor fixed. They are for reference only and are not comparable to a Clairic or Sage run (§6).

## 1. Source and revision

| What | Value |
|---|---|
| Dataset | https://huggingface.co/datasets/Extend-AI/RealDoc-Bench, revision `08dde9d20d844943fc328b161f9d332c7e8bdee4` (lastModified 2026-09-09T21:18:31Z, created 2026-05-24; not gated; HF API `downloads` = 3,722 on 23 Oct) |
| `qa_bank.json` | sha256 `70f21b74…ffcce6`, 1,094,786 bytes. It has 1,356 items and 3,742 gold keys, and it is byte-identical to the survey lane's copy |
| `manifest.json` | sha256 `ad47e62a…2be5745`. It lists 581 documents (`metadata_as_of` 2026-09-09). It holds each PDF's sha256, `ai_status`, source and rights |
| Source PDFs | `docs/<source_file>.pdf` at the same revision. We downloaded 180 of them. Each sha256 matches manifest.json, and `build.py` checks this |
| Scorer and harness code | https://github.com/extend-hq/realdoc-bench at commit `fb26a6876481de76dc293f722ab4efa71279904d` (2026-07-30, "add extend_light_v1_0_0 parser + leaderboard score (#17)"). The tarball has sha256 `c8c8bb5f…43a1`, and we unpacked it, minus the logo, in `raw/github/src/` |
| Paper | Joshi et al., *RealDocBench: A Benchmark for Field-Level QA and Layout Understanding on Real-World Regulated Documents*, arXiv 2606.07401 v1 (2026-06-05). It is at https://arxiv.org/abs/2606.07401, saved as `raw/pages/arxiv_2606.07401.{pdf,txt}` and `_abs.html` |
| Vendor blog | https://www.extend.ai/resources/realdocbench (published 2026-05-26, updated 2026-08-27), saved as `raw/pages/extend_realdocbench_blog.md` |

The question count has a discrepancy, which we record but have not resolved. The HF card and the GitHub README "Datasets" line both say "1,359 questions". The bank file, the README leaderboard header and the paper all say 1,356. We use the bank file's count of 1,356.

## 2. Licence

| Layer | Licence | Evidence | Status |
|---|---|---|---|
| QA bank and gold answers | CC BY 4.0 | The HF card has `license: cc-by-4.0`, and its "License" section says "The QA bank and gold answers are licensed under CC BY 4.0" (`raw/hf/README.md`). The manifest has `annotation_license: CC-BY-4.0` | Verified |
| Code, including `deep_equal` | Apache-2.0 | `raw/github/src/.../LICENSE`; GitHub API `license: Apache-2.0` | Verified |
| Source PDFs | Excluded from the CC BY licence, with varying rights | The HF card says "Source documents in `docs/` are excluded from this annotation license… Some source rights remain unverified." The manifest's `document_license` is null for 581/581 docs | UNKNOWN / mixed |

The manifest's rights fields break down like this:
- Across the whole set, `source_rights.status` is not_established 471, public_domain_us_federal_work 58, copyright_notice_no_reuse_license 31, explicit_license 8, agency_reuse_policy_with_conditions 8, distribution_restrictions_no_open_license 5. `ai_status` is ai_generated_edit 291, collected_real_document 196, unknown 94.
- For our 200 items, the status counts are not_established 164, public_domain_us_federal_work 21, copyright_notice_no_reuse_license 11, agency_reuse_policy_with_conditions 3, distribution_restrictions_no_open_license 1.
- This repo does not redistribute the PDFs or the rendered pages; see `LICENSE-DATA.md`.

## 3. Items and sampling (`build.py`, `build_report.json`)

1. **Field plan for every bank question.** We use the official typed template, `build_template(response_format, gold_dict)`, which is the upstream rule including `widen_types_from_gold`. Widening changed the gold on 0 questions.
2. **Eligibility.** A question is eligible if it has at least one non-boolean, non-list key, and 1,201 of 1,356 are. The 155 questions that hold only booleans or lists go to the decision side test (suite design decision C17).
3. **Allocation.** We allocate in proportion to eligible counts, using largest remainder, with N=200:

   | Domain | Eligible | Sampled |
   |---|---:|---:|
   | finance | 305 | 51 |
   | medical_healthcare | 141 | 23 |
   | mortgage | 444 | 74 |
   | supply_chain | 311 | 52 |

   In our judgement, proportional rather than equal allocation keeps the sample shaped like the published full-bank number. The cost is that medical, the hardest domain in the paper, gets only 23 questions and 37 fields, so its CI is wide.
4. **Order.** We set `rng = random.Random(20)` and take the domains in sorted order. We sort the question ids by (doc number, question number) and then apply `rng.shuffle`.
5. **Walk.** We walk each domain's order until its quota is filled. We accept a question only if its PDF's sha256 matches the manifest and `pdfinfo` reports at most 4 pages.
   - We checked page counts on all 180 sampled PDFs, and every one has 1 page. So 0 questions were dropped (0 for page count, 0 for sha mismatch). This matches the paper, which says "documents are normalized to single-page PDFs" (§3.1).
6. **Shape of the sample.**
   - The sample has 180 unique docs, with at most 3 questions per doc.
   - The bootstrap cluster is `meta.doc_id`, the source PDF. The paper's CIs are document-clustered too.
   - The number of fields per item is 1 for 63 items, 2 for 41, 3 for 26, 4 for 36, 5 for 9, 6 for 19, 7 for 1, and 8 for 5.
7. **Sealed holdout.** `random.Random(20).sample(sorted(ids), 40)` sets `meta.holdout = true` on 40 items (finance 10, medical 5, mortgage 13, supply chain 12). We send these to Sage only on the final run (suite design decision L13).
8. **Images.**
   - We render each page with poppler `pdftoppm` 26.03.0 at 150 dpi. That matches upstream `_vision_base.DEFAULT_DPI = 150`, which rasterises with PyMuPDF.
   - The output is PNG, with JPEG q90 as the fallback when a page exceeds 4 MiB or the item exceeds 8 MiB. The fallback applied to 2 pages (`finance_109`, `mortgage_135`), and 3 items use them.
   - Each item's image is p50 746 KB, p95 2.69 MB and max 3.73 MB.
   - A rebuild produced byte-identical images (180/180 sha256 equal).
   - Questions on the same doc share the image file.
9. **Text.** Every item has `text: null` and `text_source: "none"`.
   - RealDoc ships no OCR, and 127/180 docs have no text layer. Every ai_generated_edit and "unknown" doc in the sample is image-only.
   - The 53 docs with a text layer (all `collected_real_document`) carry it as `meta.alt_texts.pdf_text_layer` (`pdftotext -layout`). It is not a primary text, because the gold is reachable in it for only 80/116 non-null, non-enum fields. Some misses come from the crude matcher, for example on composite golds, so 80/116 is a lower bound.
   - `meta.pdf` points at the source PDF, so `harness/ocr.py` renders it at 200 dpi itself.

## 4. Fields

The template type hint is the source of truth, as in the upstream SYSTEM_PROMPT ("The TYPE HINT inside `< >` is the source of truth"). The gold's Python type does not decide the field type.

| RealDoc hint | Our `type` | Sage 2 / Clairic (suite design decision C2) | Bank count | In sample | Notes |
|---|---|---|---:|---:|---|
| `string` | text | text / text (span) | 1,872 | 307 | `fuzzy: true` (upstream `string_keys`) |
| `one of: a \| b …` (enum) | text | text / text | 565 | 117 | The options are in the instructions and in `enum_options`. 48 are `enum_state: true`, meaning the options are only checkbox-state words such as checked/unchecked, which the page does not print. 69 are label enums, such as yes/no, IN/OUT and Public/Private |
| `integer` | number, `integer: true` | integer / number integer | 386 | 45 | Widened to a non-integer number if the gold is a non-integral float. 17 bank keys are templated `<digits>` but hold a float gold, and 12 of those are non-integral (e.g. 67.15). 0 such keys landed in the sample |
| `number` | number | decimal / number | 484 | 89 | Not split into money, because currency cannot be assigned reliably from the key name |
| `date string YYYY-MM-DD` | date | date / date | 122 | 15 | The gold is ISO |
| `boolean` | excluded | n/a | 306 | 16 | Kept in `meta.excluded_fields` with gold, for the Clef / Sage decision side test |
| `list of strings` | excluded | n/a | 7 | 0 | |

- 39 fields have a null gold: string 28, number 5, integer 4, date 2. They fall only in mortgage (19) and supply chain (20).
  - In the whole bank, 188 of 3,742 keys are null (5.0%). They sit on 109 questions (8.0%): mortgage 107, supply chain 75, finance 5, medical 1.
  - This corrects the spec's "about 14% null golds". The 14% is 188 nulls divided by 1,356 *questions* (13.9%), not a share of fields.
- Some upstream keys are not valid field names.
  - Of the upstream keys, 191/573 break `^[A-Za-z_][A-Za-z0-9_]{0,63}$`, for example `Approved: Approved subject to the additional conditions stated below, if any.`.
  - We sanitise each one deterministically. Non-word characters become `_`, a leading digit gets an `f_` prefix, and names over 64 characters keep 55 and add an 8-hex sha1 suffix.
  - The original key is in `realdoc_key` and `meta.name_map`, and the instructions quote it.
- The instructions are identical for every arm:
  - They start with the RealDoc question, verbatim, followed by `Answer field: "<original key>".`
  - Next comes one line that paraphrases the upstream SYSTEM_PROMPT rule for that hint:
    - text: verbatim copy (rule 5);
    - enum: exactly one of the options (rule 3);
    - number: a bare number (rule 1);
    - date: YYYY-MM-DD (rule 4).
  - Last comes the null rule, "If the value is absent, blank, redacted or illegible, return null" (rules 6/6a).
  - Upstream gives the extractor only the question and the typed template, not `response_format`, so we don't pass `response_format` either. We keep it in meta.
- For re-scoring and other arms, we keep `meta.official_template`, `meta.official_gold_dict` (including booleans), `meta.official_str_keys` and `meta.question`. An LLM-JSON control arm can therefore send the upstream SYSTEM_PROMPT plus the template verbatim (`realdoc_official.SYSTEM_PROMPT`).

## 5. Metric

- The primary score is RealDoc per-field accuracy, computed by `scripts/realdoc_score.py`.
  - It applies `fuzzy_equal` to keys in `official_str_keys` and `deep_equal` to the rest, on `official_gold_dict`.
  - We inherit these upstream semantics:
    - A null prediction on a null gold is correct, and on a non-null gold it is wrong.
    - Numbers compare by `float(a) == float(b)`, after stripping `$ , %` from numeric strings.
    - Strings are normalised for smart quotes, dashes, `<br>`/`**` and OCR character spacing. They are case-sensitive, with one exception. The fuzzy path lower-cases and strips punctuation, and it accepts a rapidfuzz ratio of at least 92 only when both sides have at least 5 words.
    - Enums and dates are exact. For example, "Yes" does not equal a `yes` option token.
  - The vendored copy is verbatim, and `verify()` checks 30/30 definitions. It also passes all 27 asserts transcribed from upstream `tests/test_evaluate_scoring.py`, plus 3 null-handling asserts of ours (30/30).
- We report these numbers per arm and never pool them across sets (`realdoc_score.summarize`):

  | Measure | Definition |
  |---|---|
  | `field_acc` | Correct / 573 scored fields |
  | `field_acc_cov` | Correct / 589 gold keys, with excluded booleans counted as misses (suite design decision L9) |
  | `question_acc` | All scored fields correct |
  | `question_acc_strict` | Same, and the question had no excluded key |

  The scorer sanity check (`scripts/realdoc_score.out.txt`) gives:

  | Prediction | field_acc | field_acc_cov | question_acc | question_acc_strict |
  |---|---:|---:|---:|---:|
  | Gold | 1.000 | 0.973 | 1.000 | 0.945 |
  | All-null | 0.068 | 0.066 | 0.010 | 0.005 |

  So "always abstain" scores 6.8%.
- Report these slices:
  - domain;
  - hint (string / enum label / enum state / integer / number / date);
  - null vs non-null gold;
  - capability tags (`meta.capabilities`; the top ones in the sample are field_value_pairing 74, checkbox_state 53, column_alignment 52, table_structure 47, row_binding 43, scanned_form 28, blank_field 20);
  - `ai_status`;
  - `gold_reachable` on the harness OCR text (suite design decision C9).
- In our judgement, the scoring has these fairness caveats:
  - The page does not print the 48 `enum_state` golds (checked/unchecked). Span-returning extractors (Clairic text spans, Sage `verbatim`) can't produce them unless they deviate from span semantics. Report the slice separately, and report the head-to-head with and without it.
  - Some string golds are composites rather than one span, e.g. `finance_q332`: `"2022: (4,324), 2021: (3,397), 2020: 11,263"`.
  - A few plain `<string>` golds are also states rather than printed text, e.g. `finance_q146`: `"selected (check_mark)"`. A crude regex in the verifier's check finds 1 such gold in the sample. Upstream gives these the same rule-5 "verbatim" instruction, so our instruction is faithful, but no span extractor can return them.
  - The verifier noted a denominator difference. Upstream `summarize()` keeps only records with `ok=True`, so a call that errors out after 3 retries drops out of the denominator. A JSON that fails to parse still counts, with all its fields as misses. `realdoc_score.summarize` is stricter, because it counts every missing prediction as a miss. Report the error count per arm.
- The planning CI (`survey/selection-scripts/ci_halfwidth.out.json`, n=200, p=0.85) is plus or minus 4.9 points for a single arm and plus or minus 6.2 points for a paired difference. Use a doc-clustered paired bootstrap (seed 20, 10k resamples).

## 6. Published baselines (for reference only, and only numbers I saw on the opened pages)

Every row below has the same setting:
- Extend, the vendor, ran every row and ranks itself #1.
- The rows cover the full bank: 1,356 questions, 581 docs and 3,742 keys, including the 306 booleans and 7 lists.
- The parser varies, turning the PDF into markdown. The extractor is fixed at Gemini 3 Flash (`gemini-3-flash-preview`, temperature 0) and reads markdown only.
- Every row is scored with the same `deep_equal`.

Our arms are not comparable, for these reasons:
- They swap the extractor (Clairic or Sage 2) and use Tesseract as the parser.
- They run on a 200-question subset without booleans.
- No row below is a like-for-like comparison. The nearest legitimate anchor is a metered arm we have not run. It would feed the harness OCR to Gemini 3 Flash with the upstream prompt on these 200 items, labelled "our replication". It needs the maintainers' OK.

The paper's Table 3 gives per-field and per-question percentages. The ± figure is the 95% doc-clustered bootstrap half-width, with B=10,000. The source is https://arxiv.org/abs/2606.07401

| System (as parser) | Per-field | Per-question | Provenance |
|---|---:|---:|---|
| Extend Performance v2 | 96.0 ±0.8 | 90.9 ±1.6 | Vendor's own product |
| Claude Opus 4.8 (non-thinking) | 93.0 ±1.4 | 87.1 ±2.1 | Vendor-run |
| LlamaParse (Agentic) | 92.2 ±1.5 | 84.5 ±2.2 | Vendor-run (competitor) |
| Reducto (Agentic) | 91.4 ±1.7 | 83.8 ±2.3 | Vendor-run (competitor) |
| Extend v1 | 90.8 ±1.7 | 82.5 ±2.4 | Vendor's own product |
| Gemini 3.5 Flash | 89.3 ±1.8 | 82.2 ±2.3 | Vendor-run |
| LlamaParse | 89.2 ±2.0 | 80.8 ±2.5 | Vendor-run |
| Azure DI | 89.1 ±1.7 | 79.6 ±2.5 | Vendor-run |
| Reducto | 88.7 ±2.0 | 80.5 ±2.5 | Vendor-run |
| AWS Textract | 70.7 ±2.7 | 54.0 ±2.9 | Vendor-run |
| Chandra-2 (open source) | 86.2 ±2.4 | 78.1 ±2.7 | Vendor-run |
| olmOCR-2 | 79.5 ±2.6 | 67.9 ±3.0 | Vendor-run |
| Nanonets-OCR-s | 77.4 ±3.6 | 68.2 ±3.2 | Vendor-run |
| Docling | 71.2 ±2.9 | 54.7 ±3.2 | Vendor-run |
| dots.ocr (2-stage) | 70.6 ±3.6 | 61.4 ±3.5 | Vendor-run |
| dots.ocr (single-shot) | 65.4 ±4.0 | 57.2 ±3.6 | Vendor-run |
| Paddle-OCR-VL | 59.6 ±4.0 | 48.5 ±3.6 | Vendor-run |
| Paddle v3 | 59.2 ±3.2 | 43.1 ±3.2 | Vendor-run |
| Nemotron Nano V2 VL | 41.6 ±3.4 | 29.3 ±3.0 | Vendor-run |

The paper's Table 4 gives per-field % by domain on the full bank (Fin 378 q, Mort 478, Supply 319, Med 181). These are selected rows:

| System | Fin. | Mort. | Supply | Med. |
|---|---:|---:|---:|---:|
| Extend v2 | 92.7 | 97.5 | 97.6 | 89.8 |
| Gemini 3.5 Flash | 83.8 | 92.5 | 92.1 | 75.8 |
| Azure DI | 82.9 | 93.7 | 90.3 | 76.7 |
| AWS Textract | 68.4 | 71.6 | 77.0 | 46.9 |

Other sources give these numbers:

- The GitHub README at fb26a687 (https://github.com/extend-hq/realdoc-bench) has the same rows, from Extend Parse 2.0 at 96.0/90.9 down to AWS Textract at 70.7/54.0, plus two rows that are not in the paper:
  - Extend Parse Light 1.0: 90.5 / 82.7.
  - Mistral OCR 4: 88.8 / 81.3.
  - The commit message reports a Light rerun of 90.6 / 83.0.
- The vendor blog (https://www.extend.ai/resources/realdocbench, updated 2026-08-27) gives slightly different numbers for the same systems, e.g. Extend Performance Parse 95.7, Gemini 3.5 Flash 89.04, and AWS Textract 70.5.
  - Cite the paper with its CIs, and note that the vendor's three sources differ by up to 0.4 points.
- I found no third-party reproduction. A search result summary mentions that LlamaIndex's ParseBench reports the opposite ordering for LlamaParse vs Extend, but that is a different benchmark. I did not open it, so it is UNKNOWN.

## 7. Contamination and other risks

- **Vendor-built.** Extend ranks itself first, and the board measures parsers, not extractors. This is Tier B (secondary) in the suite.
- **Gold provenance.** The paper (§3.2) says the gold was "drafted independently by two different models, Claude Sonnet and Gemini, working from the page image". The authors kept the agreements plus some re-included disagreements, and "every retained question… is then manually verified… by a human annotator".
  - In our judgement, the gold may lean toward Claude and Gemini reading conventions. Flag this next to any Haiku (Claude-family) or Gemini control arm.
- **Synthetic content on real layouts.** The paper (§3.1, App. A) says medical and tax documents are "real, blank form templates populated with fictitious-persona data".
  - The manifest marks 291/581 docs (93/200 of our items) `ai_generated_edit`.
  - The values are fabricated by design, which also lowers memorisation risk.
- **Contamination.** In our judgement, the risk is medium-low.
  - The QA bank has been public since 2026-05-24 (HF) and the paper since 2026-06-05.
  - Clairic launched 2026-10-19, so its training could post-date the release. That is UNKNOWN.
  - Persona values are not real records.
- **Upstream template quirks.** We inherit these unchanged:
  - Some enum option lists contain a literal `one of: N/A` token, which is a parser artefact.
  - `<digits>` keys can hold decimal golds. We widen these, and the sample has 0 of them.
  - Unicode-quoted keys (e.g. `CLASS “A” CERTIFICATE`) are never fuzzy-eligible, because upstream `string_keys` regex-matches the json-escaped template. They are enums, so this has no effect.
- **Spec correction.** The spec's risk line "Datalab folded 329 docs into OmniExtractBench" refers to OEB's `extractbench` suite (LlamaIndex ExtractBench, 329 docs), not RealDoc. The OEB suites are extractbench 329, internal 202, micro1 47, longarray 42 (survey `vendor-leaderboards-2026-10-23.md` §5). We found no RealDoc suite in OEB.
- **Harness OCR bug.** While estimating tokens, we found that `harness/ocr.py` fails on every item on this machine.
  - When Python spawns Tesseract, Tesseract cannot open the temp-dir PNG by path.
  - The verifier refined this on 2026-10-23. The trigger is the `/tmp` symlink path, not Python. In a plain shell, `tesseract /tmp/x.png stdout` fails the same way, while `/private/tmp/x.png`, a workspace path and stdin all work. `tempfile` returns `/tmp/...` here, so `Path(td).resolve()` (or stdin) should fix `harness/ocr.py`.
  - Leptonica's error echoes raw PNG bytes, and `text=True` raises `UnicodeDecodeError`.
  - Passing the image on stdin works (`scripts/ocr_proxy.py`).
  - We did not fix it here, because the harness is outside this task. It must be fixed before the OCR stage runs. (Resolved on 2026-10-23: `harness/ocr.py` now resolves the temp path and decodes stderr leniently. The OCR cache for all 200 items is in `ocr/tesseract-eng-psm3-fb6/`.)

## 8. Size and tokens (o200k_base, tiktoken 0.12.0)

| Quantity (per item) | p50 | p95 | Max | Source |
|---|---:|---:|---:|---|
| `text` field | 0 | 0 | 0 | Null on all items |
| Tesseract 5.5.0 OCR of the page (harness settings: 200 dpi, `-l eng --oem 1 --psm 3`) | 698 | 1,679 | 2,577 (mean 805) | `scripts/ocr_proxy.out.json` |
| PDF text layer (60 items whose doc has one) | 418 | 1,249 | 1,807 | `scripts/selfcheck.out.json` |
| Sum of field instructions | 172 | 827 | 2,034 | Same |

- With a crude alnum-squash match on non-null, non-enum golds, the gold is reachable in the Tesseract text for 327/417 (78.4%).
  - By type, that is text 206/279, number 110/125, and date 11/13.
  - So about 22% of answerable non-enum fields are not literally present in the free OCR. Some are OCR errors and some are composite golds. This caps any OCR-text arm, Sage included.
- Enum golds appear in the OCR text at these rates: label enums 61/69, state enums 3/48. That confirms the checked/unchecked golds are not spans.
- For planning, the OCR-text arms use about 700 tokens per page (p50). Sage 2 sees the OCR plus the field definitions. Adding the two percentile rows gives a rough estimate of about 0.9k tokens per call at p50 and 2.5k at p95.
- Everything fits Sage 2's 16,384-token cap and Clairic's 65,536.
- Clairic's per-field billing re-sends the state once per field. With 573 fields over 200 items, that is about 2.9 passes per item.
- Image tokens for Clairic vision are UNKNOWN until the C1 probe runs.
- On disk, `raw/` takes 159 MB (152 MB of PDFs), `img/` 186 MB and `textlayer/` 0.3 MB. That is about 346 MB in total, under the 1.5 GB cap.

## 9. Files and how to rebuild

```
raw/hf/                 README.md, qa_bank.json, manifest.json, .gitattributes   (HF rev 08dde9d2)
raw/docs/               180 source PDFs (sha256 = manifest)
raw/github/             realdoc-bench-fb26a687.tar.gz + src/ (unpacked, never imported or executed)
raw/pages/              arXiv 2606.07401 (pdf, txt, abs html), Extend blog .md, HF API json, HF README copy
img/                    180 page images (150 dpi; 178 PNG + 2 JPEG), shared by questions on the same doc
textlayer/              pdftotext -layout for the 53 docs with a text layer (alt text only)
items.jsonl             200 items (sha256 in build_report.json)
build.py                re-creates items.jsonl from raw/ (+ --fetch to download missing PDFs at the pinned rev)
build_report.json       counts, quotas, drops, page counts, items sha256
scripts/realdoc_official.py   VERBATIM upstream template builder + scorer (generated by vendor_official.py)
scripts/realdoc_score.py      per-field / per-question / coverage-adjusted scoring for our items (+ .out.txt)
scripts/selfcheck.py          format + scorer + image + token checks (+ selfcheck.out.json)
scripts/ocr_proxy.py          $0 Tesseract proxy for token counts + reachability (+ .out.json, ocr_proxy/*.txt)
scripts/profile_bank.py       whole-bank profile with the official template builder (+ .out.txt)
scripts/prefetch.py           parallel PDF prefetch in build.py's seeded order (convenience only)
```

```
cd .
/opt/homebrew/bin/uv run --no-project python -I datasets/realdoc/scripts/vendor_official.py
/opt/homebrew/bin/uv run --no-project python -I datasets/realdoc/build.py            # add --fetch on a fresh checkout
/opt/homebrew/bin/uv run --no-project --with tiktoken==0.12.0 --with 'rapidfuzz>=3.9' python -I datasets/realdoc/scripts/selfcheck.py
/opt/homebrew/bin/uv run --no-project --with 'rapidfuzz>=3.9' python -I datasets/realdoc/scripts/realdoc_score.py
/opt/homebrew/bin/uv run --no-project --with tiktoken==0.12.0 python -I datasets/realdoc/scripts/ocr_proxy.py
```

- The tools are poppler 26.03.0 (`pdfinfo`, `pdftoppm`, `pdftotext`), Tesseract 5.5.0 and rapidfuzz 3.14.6 (resolved from `>=3.9`, as in upstream pyproject).
- We never import or execute anything from `raw/`. `vendor_official.py` copies the upstream code as text and verifies it against `raw/github/src`.

## 10. Independent verification (2026-10-23, $0)

- The verifier ran `build.py` in a temp copy, with raw/ symlinked. It reproduced `items.jsonl` byte for byte (sha256 `57ca23f6…5485`), plus all 180 images and 53 text layers.
- To check gold against raw, the verifier drew 15 items with `random.Random(7).sample`, covering 35 fields. It compared each to `qa_bank.json` `gold_dict` by value and type, along with its excluded booleans and its question text, and found 0 mismatches. It also checked three pages by eye or against the OCR: `supply_chain_q15` (the form is adversarially filled, and the gold "Payment Terms: United States" is correct), `supply_chain_q20` and `finance_q146`.
- The format check found 0 errors across 200 items and 573 fields. Names, types, null golds, numeric and date golds, enum golds within their options, and image magic bytes all pass. Images are at most 4 MiB (max 3.73 MB), each item has one image, and the PDFs exist.
- The verifier recomputed these counts from raw:
  - The bank has 1,356 questions, 3,742 keys, 188 null keys on 109 questions, 306 booleans and 7 lists.
  - The hint counts are string 1,872, enum 565, number 484, integer 386 and date 122. Of the integer-hint keys, 17 have float golds, and 12 of them are non-integral.
  - Sample rights, ai_status, the holdout split, the rename count (191), enum counts (117, 48 state) and OCR reachability (327/417; state enums 3/48) all match.
- The verifier also opened the sources:
  - The HF API shows that `main` is still `08dde9d2` (lastModified 2026-09-09) and `cardData.license` is cc-by-4.0. A fresh download of `qa_bank.json` at that revision has sha256 `70f21b74…cce6`, which matches. The file's last commit is "Fix 4 ground-truth errors in qa_bank" (2026-06-03).
  - The GitHub API shows that the licence is Apache-2.0, and commit `fb26a687` (2026-07-30) is the latest push.
  - The README leaderboard rows match. Tables 3 and 4 of the paper (arXiv 2606.07401v1) match every row quoted in §6.
  - The blog's overall numbers (95.7 / 89.04 / 70.5) match, as does its "1,356 prompts across 581 documents".
