# kleister_charity: Kleister-Charity dev-0 (all 112 long docs and 100 short ones)

We built this set on 2026-10-23 at $0 and called no metered API. The suite plan is in the suite design notes §2.8 and §8 row 9, with decisions C7, C8, C9, L13 and L14.

In short:
- The set has 212 items: every dev-0 document over 15,000 o200k tokens (all 112) plus a seed-20 random sample of 100 of the 328 shorter ones. They are UK charity annual reports and accounts, mostly scanned.
- Each item has 8 fields, 1,696 in total. Of those, 45 have a null gold. They are absent keys in Kleister's register-derived gold, so this set makes no abstention claim (C7).
- `text` is the official Textract OCR layer (`text_source: official_ocr`). The official Tesseract 4.1.1, pdf2djvu and "best" layers of the same documents are under `alt/`, so testing OCR-engine sensitivity costs $0 in OCR.
- This is the suite's long-document set, with 108/212 items over Sage 2's 16,384-token cap. Sage therefore runs a declared window rule (356 windows, at most 5 per doc). On the Textract layer, we estimate that no document exceeds Clairic's 65,536 context.
- Each item has two scoring views, which never appear in one table:
  - TYPED is the harness `exact_norm`. This is the head-to-head.
  - COMPARABLE is the Kleister GEval micro-F1 on upper-cased `key=value` labels, reweighted to dev-0. It is the only view that may sit next to the published rows, and even then only as a reference, because those rows are test-A.
- The biggest caveats are these:
  - The licence is UNKNOWN.
  - Contamination risk is high. The data has been public since 2020, and it is in Arctic-TILT's and DocLLM's training mixes.
  - The best published LAMBERT number (83.57) used an OCR layer that is not distributed (Azure CV). The same-layer reference is LAMBERT on Textract, at 82.97.

## 1. Source and revision

| What | Value |
|---|---|
| Repo | https://github.com/applicaai/kleister-charity (GitHub API metadata in `raw/pages/repo.json`) |
| Commit | `2309f4861cc49864ac653a7fa108efc2144400b3`, the master HEAD commit from 2021-08-18, "Improve scripts for fetching PDFs". There are 4 commits, and the first is `a0ddeef3` from 2020-03-05 (`raw/pages/commits.json`) |
| Files used | `README.md`, `config.txt` (GEval config), `in-header.tsv`, `dev-0/in.tsv.xz` (6.4 MB) and `dev-0/expected.tsv`, fetched from `raw.githubusercontent.com/applicaai/kleister-charity/<commit>/…` into `raw/repo/` |
| Integrity | `build.py` checks each file's git blob sha1 against the commit tree (`raw/pages/tree.json`), and 5/5 match. `in.tsv.xz` and `expected.tsv` are byte-identical to the survey pull in `extract-bench/raw/kie-docs/kleister-charity/` |
| sha256 | `dev-0/in.tsv.xz` is `f6b0dfab…96e2b8f7`, `dev-0/expected.tsv` is `5a56ca82…e5f36`, and `config.txt` is `d7964459…c43842`. The full hashes are in `build_report.json` |
| PDFs | We did not download them, because this is a text-only set. `documents/*.pdf` are git-annex symlinks. `scripts/fetch_annex_keys.py` read the 440 dev-0 symlink targets, which give each PDF's byte size and SHA-256 (`raw/annex_keys_dev0.tsv`, stored as `meta.pdf_bytes` and `meta.pdf_sha256`). The annex remote (`raw/pages/git-annex_remote.log`) is a public S3 bucket, `https://applica-public.s3-eu-west-1.amazonaws.com/kleister-charity-git-annex/<annex_key>`. One HEAD request returned 200, with the ETag equal to the md5 file name. The 212 sampled PDFs total 1.23 GB (p50 2.0 MB, max 31.5 MB) |
| Paper | Stanisławek et al., "Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts", ICDAR 2021, https://arxiv.org/abs/2105.05796 (submitted 12 May 2021; PDF and text in `raw/pages/`) |
| Official scorer | GEval, https://gitlab.com/filipg/geval (README and `MetricsMeta.hs` copied from the NDA lane's pull into `raw/pages/`). We do not run the binary |

The split sizes (README and paper Table 1) are train 1,729, dev-0 440, test-A 609.
- Test-A gold is hidden. Gonito (`https://gonito.net/challenge/kleister-charity`) timed out on 2026-10-23, as it did for the survey. So dev-0 is the only split we can score.
- The paper says dev/test documents "were manually annotated (by two annotators)", while train labels come from the Charity Commission register (§3.2).

## 2. Licence

| Layer | Licence | Evidence | Status |
|---|---|---|---|
| Kleister-Charity repo (annotations, text layers) | None stated | GitHub API `license: null`. `GET /repos/applicaai/kleister-charity/license` returns 404 (`raw/pages/license_api.json`). The README says only "The data is not treated as sensitive" | UNKNOWN |
| Source documents | Annual reports filed with the Charity Commission for England and Wales | The README "Sources" section says "downloaded from https://www.gov.uk/government/organisations/charity-commission" | Terms not checked |

For redistribution status, see `LICENSE-DATA.md`.

## 3. Items, sampling and text

We measure the strata (suite design decision C8) in o200k tokens of the primary (Textract) text:

| Stratum | Rule | dev-0 | Sampled | Holdout | `stratum_weight` | Tokens p50 / p95 / max |
|---|---|---:|---:|---:|---:|---|
| long | over 15,000 tokens | 112 (25.45%) | 112 (all) | 22 | 1.00 | 22,754 / 43,546 / 58,423 |
| short | 15,000 tokens or fewer | 328 | 100 (`random.Random(20).sample(sorted(md5s), 100)`) | 20 | 3.28 | 4,852 / 10,547 / 14,715 |
| all | | 440 | 212 | 42 | | 16,538 / 34,843 / 58,423 |

- This corrects the spec, which says "~111 long, sample 211". The figure 111 is the count on `text_best`. On the Textract layer that Sage actually reads, the count is 112 and so n = 212. The counts on the other layers are tesseract 109, djvu 109, and best 111.
- An aggregate quoted for "dev-0" must weight each item by `meta.stratum_weight` (long 1.0, short 3.28). Long and short also get separate results.
- There is a sealed, stratified 20% holdout (suite design decision L13). Within each stratum, `random.Random(20).sample(sorted(md5s), round(0.2·n))` marks 22 long and 20 short items as `meta.holdout = true`.
- The item id is `kleister_charity-<md5>`, and `meta.doc_id` is the md5 (one item per document).
- The bootstrap cluster is `meta.charity_cluster`, not `doc_id`. This comes from the verify pass on 2026-10-23. The paper (§3.2 "Dataset split") says the set holds several yearly reports per charity and splits train/dev/test by charity. In this sample, 124/212 items belong to 59 charities with 2-3 reports each, and those reports share name, number and address gold. There are 147 charity clusters. `meta.charity_cluster` is `charity-<number>`. A doc whose number key is absent borrows the number of a doc with the same name, which happens in 1 case, Kingsheart Church. The harness `cluster_of()` reads `doc_id`, so it under-clusters this set, and its CIs are too narrow until it reads `charity_cluster`. The verify pass did not make that harness change, because the harness is outside this dataset. *(Resolved 2026-10-23: `harness/common.py` `CLUSTER_KEY["kleister_charity"] = "charity_cluster"`.)*
- The holdout leaks across charities. In 20 of the 147 charities, reports fall on both sides of the seed-20 holdout. We pre-registered the holdout per document, so we did not re-draw it. Treat it as document-level, not charity-level.
- `text` is the column `text_textract` with the README escapes decoded (`\n`, `\t`, `\f`, `\\`). Per the README, this is AWS Textract, "ver. March 1, 2020", and the paper confirms the API version.
- The alternative official layers are in `meta.alt_texts`, with the harness arm suffixes `@tesseract_layer`, `@djvu_layer` and `@best_layer`:
  - `alt/tesseract/<md5>.txt` is Tesseract 4.1.1-rc1-7-gb36c, `--oem 2 -l eng --dpi 300`.
  - `alt/djvu/<md5>.txt` is pdf2djvu/djvu2hocr 0.9.8, which is the PDF's own text layer. It is empty for 10/212 sampled docs (21/440 in dev-0), which are scans with no embedded text.
  - `alt/best/<md5>.txt` is the README's combination. It is djvu unless Tesseract is at least 1,000 characters longer. It equals djvu on 353/440 dev-0 docs and Tesseract on 87.
- The o200k token counts per layer on the sample are as follows (`scripts/selfcheck.out.json`):

  | Layer | p50 | p95 | max |
  |---|---:|---:|---:|
  | Textract (primary) | 16,538 | 34,843 | 58,423 |
  | Tesseract | 16,257 | 35,068 | 57,294 |
  | djvu | 15,569 | 37,507 | 74,528 |
  | best | 16,466 | 37,507 | 74,528 |

- There are no images (`images: []`, modality `text`).
  - Per the paper, the documents average about 22 pages (61,643 pages over 2,778 docs), and UReader reports "KLC (98% > 4 pages)". So Clairic's 4-image cap rules out a whole-document vision arm.
  - `meta.page_count` is null (UNKNOWN). We did not fetch the PDFs, and no text layer has reliable page breaks. Textract has none, and Tesseract's `\f` count (median 1) contradicts the average of about 22 pages.
  - We recorded the PDF sizes, hashes and the public URL pattern (§1) so a later page-count or vision pass can fetch the exact files.
- The official `keys` column (`meta.official_keys`) lists all 8 keys on every dev-0 document. Unlike NDA, there are no per-document decoy lists here.

## 4. Fields

| Field | Type | Clairic | Sage 2 | Gold (TYPED) | Null golds |
|---|---|---|---|---|---:|
| charity_name | text | text | text | Kleister value with `_` replaced by a space, plus surface variants (below) | 0 |
| charity_number | text | text | text | Digits (437/437 dev-0 values are all digits) | 2 |
| report_date | date (`dayfirst: true`) | date | date | ISO `YYYY-MM-DD` (month-end on 421/440 dev-0 values) | 0 |
| income_annually_in_british_pounds | money (`unit: GBP`) | number, decimals 2, unit GBP | money | `"10348000.00"`-style string | 1 |
| spending_annually_in_british_pounds | money (`unit: GBP`) | Same | money | Same. One dev-0 value is negative (−3325.26) | 1 |
| address__post_town | text | text | text | Upper case, e.g. `WESTCLIFF-ON-SEA` | 6 |
| address__postcode | text | text | text | e.g. `SS0 8HX` | 7 |
| address__street_line | text | text | text | e.g. `47 SECOND AVENUE` | 28 |

- In total there are 1,696 fields: 1,651 with gold and 45 null.
- The null golds split by stratum as long 16, short 29. Across all of dev-0, the absent keys are street_line 50, post_town 18, postcode 14, income 6, spending 6, and charity_number 3.
- This is what a null gold means here:
  - Each null field carries `null_semantics: "key_absent_in_kleister_gold"`.
  - Kleister's values come from the Charity Commission register, and annotators checked them manually on dev/test. A missing key therefore says the register or the annotators gave no value. It does not prove the document lacks one.
  - Under the harness, a null-gold field scores `correct_abstain` or `hallucinated`. Report those 45 apart, and do not use them as abstention evidence.
- The field instructions paraphrase the README's definitions (verbatim strings in `build.py` `FIELDS`).
  - The address fields say "the charity's own address … not an auditor, bank, adviser".
  - The money fields say "If the accounts are stated in thousands of pounds, give the full amount". The paper notes that about 5% of amounts need this scale inference.
  - The instructions take no examples from gold and say nothing about null. The harness system rule covers abstention.
- The surface variants are pre-registered and derived from gold only, with no hand edits:
  - Each text gold also accepts every string in any of the 4 official layers whose case-folded alphanumerics equal the gold's, with at most 3 non-alphanumerics between characters. This is the NDA party rule, so only punctuation, spacing and case can differ. Examples are `SS08HX` for `SS0 8HX`, and `Cheltenham Ladies' College`.
  - For charity_name only, `Ltd.` and `Limited` are interchangeable, because the paper (§3.2) says these name differences were normalised.
  - A field has at most 4 values. 148/1,017 text golds gained a variant (`build_report.json`).
  - Without the Ltd. rule, 13 of 29 unreachable names were exactly this case (`scripts/explore_name_reach.py`).

## 5. Metrics

TYPED, the head-to-head view, uses the harness `score.py` `exact_norm`:
- text uses `norm_text`;
- money matches within plus or minus 0.005 after `parse_number`;
- date uses `parse_date(dayfirst=True)`;
- abstention rules (a) and (b) apply.

`scripts/selfcheck.py` runs these checks, and all of them pass:

| Check | Result |
|---|---|
| Every acceptable gold value scored as a prediction | 1,858/1,858 = 1.0 |
| Wrong values | 0 |
| Null gold and a null prediction | `correct_abstain` |
| Null gold and a value | `hallucinated` |
| Money as a float and as `£10,348,000.00` | 422/422 equal gold |
| Date as `31 March 2016` and `31/03/2016` | 212/212 equal gold |
| Request bodies | Clairic sends 8 fields (money as `number/GBP`), Sage sends 8 (money type `money`), and OpenRouter requires all 8 |
| Strata, seed-20 sample and holdout | Re-derived independently from `raw/`, and identical |
| Sage window counts | Re-computed with the harness's own `windows()`, with 0 mismatches |

COMPARABLE is `comparable/kleister_geval.py`.
- It re-implements GEval `MultiLabel-F1:u`, the first metric in `config.txt`. That metric is micro F1 over upper-cased `key=value` labels across the set.
- Its per-group rows mirror config.txt's `m<…>` selections (address, money, town, postcode, street, name, number, income, spending, date).
- It reports:
  - dev-0-reweighted micro-F1 (counts multiplied by `stratum_weight`), which is the number to quote;
  - unweighted micro-F1 on the sample;
  - long and short separately;
  - true-case F1.
- We wrote the output adapter, and it is the same for every arm. It turns dates into ISO day-first, money into `%.2f`, and spaces and colons in text into `_`. It keeps commas and trailing periods, because the gold keeps them: 59 of the 3,423 dev-0 values are names ending in `Ltd.`
- It is not the GEval binary. We chose how to handle duplicate labels and per-key case.

The self-test output is in `comparable/selftest.out.json`:

| Input | GEval-style F1 (UC, dev-0-reweighted) |
|---|---:|
| expected.tsv against itself | 100.00 |
| Empty output | 0.00 |
| TYPED gold (Kleister form) through the adapter | 100.00 |
| Money as floats, dates as DD/MM/YYYY, through the adapter | 100.00 |
| Verbatim-span ceiling, from a perfect copier of Textract spans (the Kleister form if present in the text, else the most frequent surface variant) | 97.38 (name 83.05, postcode 98.73, street 98.37) |
| Gold plus an invented value for every null key | 98.40 (P 96.84) |

- So on the COMPARABLE view, arms that return spans lose up to about 2.6 points from formatting alone, and Clairic and Sage both return verbatim spans. Most of the loss is on charity_name (`Limited` versus `Ltd.`, and apostrophes).
- The TYPED view accepts those variants.

## 6. OCR-engine sensitivity: is the gold reachable in each layer? (suite design decision C9, $0)

`meta.gold_reachable_by_layer[layer][field]` records whether some acceptable gold value is visible in that layer:
- for text, as a substring after `norm_text`;
- for money, as the amount with or without thousands separators, or, separately, as a £000 figure;
- for dates, as any day/month/year surface form that parses to the gold.

The rates below are over the non-null golds of the 212 items:

| Field | Textract (primary) | Tesseract | djvu | best |
|---|---:|---:|---:|---:|
| charity_name | 92.5% | 92.9% | 87.3% | 92.5% |
| charity_number | 97.6% | 97.1% | 91.4% | 96.7% |
| report_date | 97.2% | 97.6% | 91.5% | 97.6% |
| income (exact or £000) | 86.7% | 81.0% | 69.7% | 74.9% |
| spending (exact or £000) | 89.6% | 83.9% | 73.5% | 79.1% |
| post_town | 97.6% | 98.1% | 92.7% | 98.1% |
| postcode | 76.6% | 81.5% | 72.7% | 78.5% |
| street_line | 90.8% | 90.8% | 84.8% | 90.8% |
| All 1,651 golds | 91.1% | 90.4% | 82.9% | 88.5% |

- Textract beats Tesseract on money (+5.7 pts on both income and spending) but loses on postcodes (−4.9). That makes the `@tesseract_layer` arm a real OCR-engine test, not a near-duplicate.
- On Textract, reachability is 92.7% for long items and 89.2% for short ones.
- To read `meta.gold_reachable` for money, note that `gold_reachable[f]` covers the exact amount only. The rates above, and the reachable subset, count `gold_reachable[f] or gold_reachable[f + "__thousands"]`. Use that OR when filtering, or the £000 cases drop out.
- Postcodes are an OCR-character test (verify pass, $0). Of the 48 non-null postcodes not visible in the Textract layer, 27 appear once the usual OCR confusions are allowed (0/O, 1/I/L, 5/S, 8/B), for example `HDI 6PG` for `HD1 6PG`. A span-copier returns the OCR form and scores 0 under TYPED, while an arm that repairs it scores 1. That is a real OCR-robustness signal, not a gold error.
- The document writes the amount exactly for only about 65% of money golds. Another 22% or so appear only as £000 figures (e.g. 10,348 for 10,348,000.00). For the rest, the document's figure differs from the register's, for example text `107711` versus gold `107771`, and one report is in US$000. The spot checks are in `scripts/check_money_reach.py`.
- In our judgement, the roughly 9% of golds that no layer shows verbatim cap every text-copying arm. Report every head-to-head on both the full set and the `gold_reachable` subset.

## 7. Long documents and windowing (declared harness adaptation)

| | Value (o200k estimate; neither vendor's tokenizer is known) |
|---|---|
| Items over Sage 2's 16,384 cap (content plus 347 field-def tokens) | 108/212. The other 4 long items would fit under the cap, but the harness's window of 15,000 tokens minus the defs still splits them in 2. Also, 1 short item, at 14,715 tokens, needs 2 windows because of the defs |
| Sage windows (harness `windows()`, at most 15,000 tokens minus the defs, 200-token overlap) | 356 in total. Long items have p50 2, max 5 |
| Sage rule | Take the best non-null answer by probability across windows, or else the most confident null (`harness/clients.py call_sage2`). Ranking by probability across windows is an adaptation. Sage's probabilities are not calibrated across windows (UNKNOWN) |
| Over Clairic's 65,536 context (Textract) | 0/440 dev-0 at o200k, and 0/440 at the harness's ×1.12 Clairic-token assumption. The largest doc (`…871b94cb`, 58,423 tokens, about 65.4k at ×1.12) sits at the limit. Log Clairic's `truncated` flag |
| Correction to the spec ("~0.9% exceed Clairic's 65k") | That figure holds for the djvu and best layers: 2/440 at o200k, 5/440 (1.1%) at ×1.12 (`scripts/clairic_ctx_check.out.json`). It is 0 on the primary Textract layer |
| Harness note | `harness/clients.py` sets `CLAIRIC_WINDOW = 45000`, so the "whole-doc" `clairic_text` arm still splits 5/212 items (over 45k tokens) into 2 windows. If the plan means "Clairic whole-doc" literally, raise that window for this set. The `clairic_text_w15k` arm is the windowed control (350 calls). *(Resolved 2026-10-23: the `clairic_text_whole` arm uses one 58,500-token window, `clients.CLAIRIC_WHOLE`.)* |

## 8. Published baselines (reference only; none is zero-shot on our split)

We read every number below on the linked page or PDF (saved in `raw/pages/`). All are F1 on Kleister-Charity, mostly on test-A, which has hidden gold. Ours is dev-0.

| System | Score (F1) | Setting | Text layer / input | Split | URL |
|---|---:|---|---|---|---|
| Human (annotator agreement, 100 random docs) | 97.45 | Table 3 | n/a | test-A | https://arxiv.org/abs/2105.05796 |
| LAMBERT (base) | 83.57 | Fine-tuned pipeline, mean of 3 runs (Table 3). Per entity: post_town 83.70, postcode 90.37, street_line 74.30, charity_name 77.83, charity_number 95.80, income 74.70, report_date 96.80, spending 74.20 | Azure CV 3.0.0 OCR (not distributed) | test-A | https://arxiv.org/abs/2105.05796 |
| LAMBERT on Textract (our primary layer) | 82.97 ±0.21 | Table 4 | Textract (same engine and version as `text`) | test-A | https://arxiv.org/abs/2105.05796 |
| LAMBERT on Tesseract (our alt layer) | 81.50 ±0.07 | Table 4 | Tesseract 4.1.1 | test-A | https://arxiv.org/abs/2105.05796 |
| Flair / BERT / RoBERTa / LayoutLM, Textract | 78.03 / 73.30 / 80.08 / 80.23 | Table 4 | Textract | test-A | https://arxiv.org/abs/2105.05796 |
| Flair / BERT / RoBERTa / LayoutLM, Azure CV | 81.17 / 78.33 / 81.50 / 81.53 | Table 3 | Azure CV | test-A | https://arxiv.org/abs/2105.05796 |
| Autotagger (regex, span level) | 78.16 | Table 3 | n/a | test-A | https://arxiv.org/abs/2105.05796 |
| Arctic-TILT (<1B) | 88.1 | Fine-tuned. Its SFT mix includes Kleister Charity plus 161 private charity annual reports (App. B). It cites LAMBERT 83.6 as the prior SOTA (Table 3) | UNKNOWN | Not stated (UNKNOWN) | https://arxiv.org/abs/2408.04632 |
| DUE T5 / T5+2D / T5+U / T5+2D+U | 74.3 / 72.6 / 76.0 / 75.9 | Fine-tuned, mean of 3 runs (Table 3). "Cannot process an entire document … consumed only a part" | DUE OCR | DUE test. Its sizes (1.7k/0.4k/0.6k docs) match Kleister's 1,729/440/609. In our judgement, these are the same splits. DUE hosts the data at https://duebenchmark.com/data | https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/069059b7ef840f0c74a814ec9237b6ec-Abstract-round2.html |
| DUE T5+2D+U by max input length 1,024 / 2,048 / 4,096 / 6,144 | 56.6 / 66 / 73.2 / 75.9 | Table 4. Truncation costs about 19 points | DUE OCR | DUE test | Same as above |
| GPT-4 + OCR, zero-shot | 45.9 | DocLLM's authors (JPMorgan AI Research) ran it with the per-key prompt `"{document} What is the value for the \"charity number\"?"`. The OCR engine, GPT-4 context handling and truncation are not stated (UNKNOWN) | OCR text, no layout | "test split (or dev split when test split is unavailable)", which is UNKNOWN for KLC | https://arxiv.org/abs/2401.00908 |
| Llama2-7B + OCR, zero-shot | 27.8 | Same (Table 5) | OCR text | Same | https://arxiv.org/abs/2401.00908 |
| DocLLM-1B / DocLLM-7B | 58.9 / 60.3 | Instruction-tuned including KLC train (SDDS, Table 5). DocLLM's context is 1,024 tokens | OCR text and boxes | Same | https://arxiv.org/abs/2401.00908 |
| DocLLM-1B / DocLLM-7B, KLC held out | 40.1 / 49.9 | STDD (Table 6), with KLC excluded from tuning | OCR text and boxes | Same | https://arxiv.org/abs/2401.00908 |
| mPLUG-DocOwl / UReader | 30.3 / 32.8 | Instruction-tuned on KLC train, OCR-free (DocLLM Table 5, and UReader Table 1). UReader says "OCR-Free models only input the first page" | Page image | Test split (DUE) | https://arxiv.org/abs/2310.05126 |
| DocOwl-1.5 / DocOwl-1.5-Chat | 37.9 / 38.7 | OCR-free (Table 3) | Page images | Test split (DUE) | https://arxiv.org/abs/2403.12895 |

How to read these rows:
- The closest-protocol anchors for our arms are:
  - GPT-4+OCR at 45.9, a zero-shot LLM over OCR text with per-key questions. Its split and truncation are UNKNOWN.
  - LAMBERT-on-Textract at 82.97, which uses the same OCR engine and version but is fine-tuned and on test-A.
- Quote our COMPARABLE dev-0-reweighted F1 next to them only with "dev-0 vs test-A; zero-shot vs fine-tuned".
- This corrects the spec, which lists "LAMBERT 83.57" as the reference. That row used Azure CV OCR, which the repo does not ship. The same-layer row is 82.97 (Textract) or 81.50 (Tesseract).
- The DUE table shows that input truncation matters, with 56.6 at 1,024 tokens versus 75.9 at 6,144. This is why we report the long stratum and windowing separately.
- We found no zero-shot result for a modern LLM or extraction API on Kleister-Charity dev-0. The Haiku 5.5 JSON arm is the LLM control.

## 9. Contamination risk: high

- The dev-0 gold and all four text layers have been public on GitHub since 2020-03-05. The `dev-0/expected.tsv` and `in.tsv.xz` blob SHAs in the first commit `a0ddeef3` are identical to the pinned ones (`raw/pages/tree_first_commit.json`). The paper has been on arXiv since May 2021.
- DUE (2021) redistributes KLC.
- Arctic-TILT includes Kleister Charity in its SFT data, and DocLLM instruction-tuned on KLC train.
- In our judgement, any LLM from 2024 or later, Clairic or Sage could have seen the repo.
- The survey's L1 check found that Kleister is not among the tasksource extraction-corpus sources.
- Read results as an "old public set" (L18).

## 10. Cost (list-price estimate, `scripts/estimate.out.txt`, from `harness/estimate.py`; nothing was run)

| Arm | Calls | Input tokens | Estimate |
|---|---:|---:|---:|
| clairic_text (whole doc, 45k harness window) | 217 | 30.4M (8 fields, each re-billing the state, ×1.12) | about $3.04 |
| clairic_text_w15k (windowed control) | 350 | 30.6M | about $3.06 |
| sage_precise (windowed) | 356 | 3.56M in and 0.47M out (assuming 164 out-tokens per field) | price not public |
| haiku_json | 212 | 3.53M | about $0.38 |

An `@tesseract_layer` repeat of any arm costs about the same again.

## 11. Risks and limits

- The licence is UNKNOWN (§2).
- Test-A gold is hidden. Every published number is on test-A or an unstated split, so each one is a reference only.
- The null golds are not abstention evidence (§4).
- Gold comes from the register. About 9% of golds are not visible verbatim in any layer (§6), so the scorer can mark a correct reading of the document as wrong. Use the `gold_reachable` subset.
- Windowing is a declared harness adaptation for Sage. It also changes the cost and the evidence each call sees. Clairic gets both whole-doc and windowed arms so the effect can be separated.
- o200k is a proxy. Neither Clairic's nor Sage's tokenizer is known, so the 15k stratum boundary and the 65k check are estimates.
- The COMPARABLE scorer is a re-implementation, checked only by its self-test.
- The set is text only. We did not fetch the 1.23 GB of sampled PDFs, page counts are UNKNOWN, and there is no vision arm.
- The set has n = 212 overall, and the short stratum is n = 100 with weight 3.28. The dev-0-reweighted CI is wider than the unweighted one. Use the bootstrap with weights, clustered on `meta.charity_cluster` (147 clusters), or report the strata apart.

## 12. Files and rebuild

```sh
cd ..
/opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/kleister_charity/build.py                 # items.jsonl, alt/, build_report.json (no network)
/opt/homebrew/bin/uv run --no-project --with tiktoken --with httpx python -I extract-bench/datasets/kleister_charity/scripts/selfcheck.py
python3 -I extract-bench/datasets/kleister_charity/comparable/kleister_geval.py --selftest
python3 -I extract-bench/datasets/kleister_charity/comparable/kleister_geval.py --preds-dir extract-bench/runs/RUN/preds/ARM/kleister_charity
```

- `build.py` uses the stdlib and tiktoken, and it never downloads anything.
- `scripts/fetch_annex_keys.py` is a one-off, free GitHub fetch of the PDF annex keys. Its output, `raw/annex_keys_dev0.tsv`, is optional input to `build.py`.
- `scripts/explore.py`, `explore_gold.py`, `explore_name_reach.py`, `check_money_reach.py` and `clairic_ctx_check.py` (with `.out.json`) are for exploration only. `scripts/build.out.txt` is the last build's stdout.
- `raw/repo/` holds the pinned repo files.
- `raw/pages/` holds the pages and papers we opened:
  - GitHub API JSON, the git tree and the annex `remote.log`;
  - arXiv abstracts, PDFs and text for 2105.05796, 2408.04632, 2401.00908, 2310.05126 and 2403.12895;
  - the DUE NeurIPS 2021 PDF;
  - the GEval README and MetricsMeta.
- raw/ is 32 MB (about 25 MB of it papers), alt/ is 41 MB, and items.jsonl is 15 MB. The dataset directory is about 88 MB.

## 13. Independent verification (2026-10-23, $0)

| Check | Result |
|---|---|
| (a) `build.py` re-run in a temp copy of `raw/` | `items.jsonl`, `build_report.json` and `alt/` are byte-identical. After the fix below, a re-check found them identical across two rebuilds |
| Raw files against GitHub | We re-downloaded `dev-0/expected.tsv` and `dev-0/in.tsv.xz` from `raw.githubusercontent.com/applicaai/kleister-charity/2309f486…/`, and they are byte-identical. We also re-checked live on the GitHub API the commit list (4 commits, HEAD 2309f486 2021-08-18, first a0ddeef3 2020-03-05), `license: null` and the `/license` 404 |
| (b) 15 random items (seed 7), 120 fields against `expected.tsv` | 0 mismatches. Every text variant has the gold's alphanumerics (or the Ltd./Limited swap). Some variants are OCR splits such as `br un el Care` for Brunelcare, which the stated rule allows |
| Harness scoring of raw gold | 1,651/1,651 raw gold values (money as `£x,xxx.xx`, dates as DD/MM/YYYY) score 1. 0/1,582 other-document values score above 0. 0/422 £000-scale money answers score above 0. `validate_item` finds 0 problems |
| (c) Format | 212 items, 1,696 fields, 45 null golds, all names valid, types text/date/money, `images: []`, `text_source: official_ocr` (Textract is the dataset's own OCR layer) |
| (d)/(e) Licence, revision, baselines | We re-read in the saved papers every Table 3/4 number and the Arctic-TILT 88.1 / 83.6, DUE 74.3/72.6/76.0/75.9 and 56.6/66/73.2/75.9, DocLLM 45.9/27.8/30.3/32.8/58.9/60.3 and 40.1/49.9, UReader 32.8 and DocOwl-1.5 37.9/38.7 rows. All 8 URLs return 200 with the right titles |
| (f) Mapping | GEval MultiLabel-F1 is micro over labels (`geval_MetricsMeta.hs`), as the COMPARABLE scorer implements. This card states the TYPED deviations (variants, Ltd./Limited, per-field exact_norm, nulls reported apart) |
| Fix applied | The verifier added `meta.charity_cluster` (and a note) in `build.py` and rebuilt, and no other field changed. It also added the money-reachability and postcode-OCR notes in §6 |

