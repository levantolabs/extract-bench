# kleister_nda: Kleister-NDA dev-0 (all 83 EDGAR NDAs)

We built this set on 2026-10-23 at $0 and called no metered API. The suite plan is in the suite design notes §2.7 and §8 row 8, with decisions L13, C2 and C7.

In short:
- The set includes all 83 dev-0 documents, with no sub-sampling. They are born-digital US non-disclosure agreements taken from SEC EDGAR exhibits, 1 to 17 pages each.
- Every item has 5 fields, 415 in total. Of those, 75 have a null gold (absent keys): effective_date 21, jurisdiction 5, term 49.
- The fields are fixed contract fields that map to both APIs' types: a date, a duration (the suite's only duration source, n=34) and text. The set also includes the parties as a list.
- `text` is the official born-digital text layer, so `text_source` is `native`. The official Tesseract and Textract OCR layers of the same documents are under `alt/` for an OCR-noise arm.
- Each item has two scoring views, which never appear in one table:
  - TYPED uses the harness `exact_norm` and `list_f1` scorers. This is the head-to-head.
  - COMPARABLE uses the Kleister GEval metric (micro F1 on upper-cased `key=value` labels). Our re-implementation is in `comparable/`.
- LayoutLMv2 published its Kleister-NDA numbers on this same 83-document validation split (dev-0). That makes them the closest-protocol reference, although those models were fine-tuned and ours run zero-shot.

## 1. Source and revision

| What | Value |
|---|---|
| Repo | https://github.com/applicaai/kleister-nda (GitHub API metadata saved in `raw/pages/repo.json`) |
| Commit | `c2c7bf069b919bfb618b268fda2a2c079c0db316` (master HEAD, 2021-08-18, "Format results as percentages"). There are only 2 commits, and the first is `6dbaf807` from 2020-03-05 (`raw/pages/commits.json`) |
| Files used | `README.md`, `config.txt` (GEval config), `in-header.tsv`, `dev-0/in.tsv.xz`, `dev-0/expected.tsv`, and the 83 dev-0 PDFs `documents/<md5>.pdf` (5.4 MB). We fetched them from `raw.githubusercontent.com/applicaai/kleister-nda/<commit>/…` into `raw/repo/` |
| Integrity | `build.py` checks every file's git blob sha1 against the commit tree (`raw/pages/tree.json`), and 5/5 files and 83/83 PDFs match. Each PDF's MD5 equals its file name (83/83) |
| sha256 | `dev-0/in.tsv.xz` is `728e6b4c…6eb48`, `dev-0/expected.tsv` is `2ac328d8…dca34`, and `config.txt` is `377b2143…2e15b` (full hashes in `build_report.json`). These files are byte-identical to the earlier survey pull in `extract-bench/raw/kie-docs/kleister-nda/` |
| Paper | Stanisławek et al., "Kleister: Key Information Extraction Datasets Involving Long Documents with Complex Layouts", ICDAR 2021, https://arxiv.org/abs/2105.05796 (v1, 12 May 2021; PDF and text in `raw/pages/`) |
| Official scorer | GEval, https://gitlab.com/filipg/geval (master `a7f81c5a`, 2022-11-18; README and `src/GEval/MetricsMeta.hs` in `raw/pages/`). We do not run the binary, because we do not execute downloaded code |

The split sizes (README and paper) are train 254, dev-0 83, test-A 203. Test-A gold is hidden, so dev-0 is the only split we can score. The README marks it "hidden", and the survey found the Gonito host unreachable.

## 2. Licence

| Layer | Licence | Evidence | Status |
|---|---|---|---|
| Kleister-NDA repo (annotations, text layers, PDFs) | None stated | GitHub API `license: null`. `GET /repos/applicaai/kleister-nda/license` returns 404 (`raw/pages/license_api.json`). The README has no licence text, and the paper says only "We share the datasets to encourage progress…" | UNKNOWN |
| Source documents | SEC EDGAR filings (public record) | The README "Sources" section says "Original data was gathered from the Edgar Database" | Terms not checked |

For redistribution status, see `LICENSE-DATA.md`.

## 3. Items and text

- The set has n = 83 items, the whole dev-0 split, in `in.tsv` order (sorted by MD5). The item id is `kleister_nda-<md5>`. `meta.doc_id` is the md5, and it is the bootstrap cluster (one item per document).
- The build uses seed 20 only for the sealed 20% holdout (suite design decision L13). `random.Random(20).sample(sorted(ids), round(0.2*83))` marks 17 items with `meta.holdout = true`.
- `text` is the `text_best` column with the README escapes decoded (`\n`, `\t`, `\f`, `\\`).
  - `text_best` is identical to `text_djvu` on 83/83 documents. This is the pdf2djvu/djvu2hocr 0.9.8 text layer of the born-digital PDF. Per the README, the PDFs were made with Puppeteer from the original EDGAR HTML. That makes the text `native`, not OCR.
  - No layer has any `\f` page breaks.
- The alternative layers are in `meta.alt_texts`, with the harness arm suffixes `@tesseract_layer` and `@textract_layer`:
  - `alt/tesseract/<md5>.txt` is Tesseract 4.1.1, `--oem 2 -l eng --dpi 300` (README).
  - `alt/textract/<md5>.txt` is AWS Textract, the "March 1, 2020" API version.
  - The paper's Table 4 publishes baselines for each layer (§6), so an OCR-noise comparison has reference rows.
- There are no images (`images: []`, modality `text`). We keep the PDFs in `raw/repo/documents/` for a possible later vision arm. Page counts come from `pdfinfo`: p50 5, p95 11.9, max 17, 489 pages in total. Most documents exceed Clairic's 4-image cap, so a vision arm would need a page-selection rule.
- The o200k token counts (tiktoken, `scripts/selfcheck.out.json`) are as follows:

  | Layer | p50 | p95 | max | total |
  |---|---:|---:|---:|---:|
  | text_best (primary) | 3,096 | 6,128 | 8,783 | 281,020 |
  | tesseract | 3,124 | 6,113 | 8,769 | 280,732 |
  | textract | 3,157 | 6,122 | 8,745 | 283,435 |

  - The content plus the Sage field definitions comes to a maximum of about 8.9k o200k tokens, so 0/83 items exceed Sage 2's 16,384 cap and no windowing is needed.
  - Sage's own tokenizer is UNKNOWN, so this is an o200k estimate.
- We keep the official `keys` column in `meta.official_keys`.
  - For each document, the official task lists the keys to predict, and these include "decoy" keys that have no value.
  - The column is not a reliable schema. In 74 documents it lists all four keys, but 3 list only `jurisdiction` even though their gold has a `party`.
  - We ask all five fields on every document, the way a real user sends a fixed schema. Every key absent from `expected.tsv` becomes a null gold.
  - The verifier added a note on 23 Oct. Of the 75 null golds, 14 (effective_date 6, term 8, on 9 documents) are on keys that the official `keys` column does not ask for that document. Kleister never posed those questions, so "absent" is our inference and not an annotation. Spot-checks find plausible values in some of them, e.g. `54589bbc` term "for a period of twelve (12) month[s]" and `64ee806e` term "for a period of three (3) years". `meta.null_outside_official_keys` lists these fields, and `n_null_outside_official_keys` in `build_report.json` gives the count. Exclude them from the C7 abstention slice unless they are hand-audited. The officially asked nulls are 61: effective_date 15, jurisdiction 5, term 41.

## 4. Fields

| Field | Item type | Clairic | Sage 2 | Gold (TYPED) | Null golds | Headline |
|---|---|---|---|---|---:|---|
| effective_date | date (`dayfirst: false`) | date | date | ISO `YYYY-MM-DD` from expected.tsv | 21 | yes |
| jurisdiction | text | text | text | Kleister value with `_` replaced by a space (21 distinct US states) | 5 | yes |
| term | duration | duration, with `unit: "months"` for every item | duration | ISO 8601 from Kleister `{n}_{units}`, so `2_years` becomes `P2Y` and `12_months` becomes `P12M` | 49 | yes |
| party | list (items text) | list | n/a (unscored) | All gold parties, given as a list of acceptable lists (see below) | 0 | no (list-capable arms) |
| party_any | text | text | text | Any gold party, or one of its surface forms | 0 | no (reported separately) |

- In total there are 415 fields: 340 with gold and 75 null. There are 160 party values (69 documents with 2 parties, 4 with 3, the rest with 1).
- The term values and their counts are `2_years` 13, `3_years` 7, `1_year` 6, `12_months` 3, `5_years` 2, and one each of `10_years`, `9_months` and `24_months`.
- The Clairic unit is fixed per field, not per item, so it does not leak the gold unit. The harness scores durations without regard to unit, within 2% (12 months equals 1 year).
- The field instructions paraphrase the README definitions (verbatim strings in `build.py` `INSTR`).
  - They don't mention null. The harness system rule covers abstention.
  - The jurisdiction instruction says "state or country name only, e.g. New York".
- The type map follows suite design decision C2. Date, duration and text map 1:1. `list` has no Sage type, so Sage's `party` field is unscored and `party_any` is its stand-in.

Party surface variants follow a pre-registered rule, derived from gold only.
- Kleister normalisation drops commas and title-cases names. For example, the text "JDA Software Group, Inc." has the gold "Jda Software Group Inc.".
- Only 86/160 gold parties appear verbatim (case-insensitive) in the text, and 158/160 do after squashing to alphanumerics (`scripts/explore_gold.py`).
- So each gold party accepts two kinds of value:
  - the Kleister value;
  - every text span whose alphanumerics equal the gold's (at most 3 non-alphanumerics between characters, and at most 4 forms after `norm_text` dedupe).
- The `party` gold is the product of these per-party variants, given as a list of acceptable lists (max 4 combinations) and scored by multiset F1.
- The `party_any` gold is the union of the variants.
- Of the gold parties, 157/160 have a surface form. We keep the other 3 unchanged, and `build_report.json` flags them:
  - for `Ii-Vi Inc.`, the text says "Incorporated";
  - for `Seawell Ltd.`, the text says "Seawell Limited";
  - for `Oceanfinancial Corp.`, the text says "OceanFirst Financial Corp.", so the gold appears to drop "First". In our judgement, this is an annotation error.
- party_any is reachable verbatim on 82/83 documents, and jurisdiction on 78/78.

## 5. Metrics

TYPED is the head-to-head. It uses the harness `score.py`: `exact_norm` (date via `parse_date`, duration via `parse_duration` with a 2% tolerance, text via `norm_text`), `list_f1` for `party`, and abstention rules (a) and (b).

`scripts/selfcheck.py` gives these results:

| Check | Result |
|---|---|
| Every acceptable gold value scored as a prediction | 582/582 = 1.0 |
| Wrong value | 0 |
| Null gold and a null prediction | `correct_abstain` |
| Kleister-form `{n} {units}` and Clairic-style numeric months for term | 34/34 equal gold |
| Kleister-form party lists | 83/83 = 1.0 |
| Request bodies | Clairic sends 5 fields (term as `duration/months`), and Sage sends 4 (party dropped) |

COMPARABLE is the only view that may sit next to the published rows. Its scorer is `comparable/kleister_geval.py`.
- It re-implements GEval `MultiLabel-F1:u`, the headline metric in `config.txt`. That metric is micro F1 over the whole set on upper-cased `key=value` labels. GEval's own description says it is "calculated for the whole test set … a micro-average".
- Its per-key rows mirror the `m<key=…>` selections in `config.txt`.
- It is not the GEval binary. We chose how to handle duplicate labels and case in the per-key filter, and GEval's behaviour on both is UNKNOWN.
- We wrote the output adapter, and it is the same for every arm:
  - It turns dates into ISO.
  - It removes commas. Kleister gold has 0 commas in its 334 values.
  - It turns spaces and colons into `_`.
  - It writes term as `{n}_{units}` in the arm's own unit and never converts it.
  - For an arm without list output, it uses that arm's single `party_any` answer.

The self-test output is in `comparable/selftest.out.json`:

| Input | GEval-style F1 (UC) |
|---|---:|
| expected.tsv scored against itself | 100.00 |
| Empty output | 0.00 |
| TYPED gold through the adapter | 100.00 |
| Text surface forms (with commas) through the adapter | 98.50 |
| Sage-like ceiling, with perfect answers but one party per document | 86.97 (party R 51.88) |
| Clairic term ceiling with `unit: months`, with perfect values in the months unit | term F1 14.71 (5/34) |

- So on the COMPARABLE view, Sage 2 is capped by the missing list type and Clairic by its unit choice on term, because Kleister keeps the document's original unit.
- Report these caps next to any COMPARABLE number. The TYPED view has neither cap.
- A verifier note from 23 Oct says that `harness/score.py` does not read the per-field `headline: false` flag or `meta.headline_fields`. A plain `aggregate()` therefore pools `party` and `party_any` with the three headline fields. Filter to `meta.headline_fields` when reporting the headline TYPED number. *(Resolved 2026-10-23 by the harness integration pass: `score_item` now reads `meta.headline_fields` and `headline: false`, `aggregate()` reports a separate `headline` block, and `run.py score` pairs use headline fields only.)*
- A critic note from 2026-10-23 records that `harness/score.py` now also parses contract-style predicted dates such as "15th day of January, 2019" (`_pred_date`). Before that change, a correct verbatim effective-date span in that form scored 0. Gold parsing is unchanged.

## 6. Published baselines

All rows are fine-tuned on Kleister-NDA train, not zero-shot. A web search found no zero-shot LLM score on Kleister-NDA, so the Haiku 5.5 JSON arm is the LLM control.

| System | Score (F1) | Setting | Split | URL |
|---|---:|---|---|---|
| LayoutLMv2-LARGE | 85.20 | Fine-tuned, with entity-level F1 "using the official evaluation toolkit" (Table 9) | dev-0 (validation, same 83 docs as ours) | https://arxiv.org/abs/2012.14740 |
| LayoutLMv2-BASE | 83.30 | Same | dev-0 | https://arxiv.org/abs/2012.14740 |
| LayoutLM-LARGE / BASE | 83.40 / 82.70 | Same | dev-0 | https://arxiv.org/abs/2012.14740 |
| UniLMv2-LARGE / BERT-LARGE / BERT-BASE | 81.80 / 79.10 / 77.90 | Same | dev-0 | https://arxiv.org/abs/2012.14740 |
| RoBERTa-BASE (as cited from Graliński et al. 2020) | 79.30 | Cited in LayoutLMv2 Tables 2 and 9. The verifier found on 23 Oct that 79.30 does not appear in Graliński et al. 2020 (https://arxiv.org/abs/2003.02356). That paper's RoBERTa-base Kleister-NDA ALL is 77.07 in v2 (Table 5, test set) and 74.2 in v1. The cited number's split and provenance are UNKNOWN, so don't quote it as a dev-0 number | UNKNOWN | https://arxiv.org/abs/2012.14740 |
| LAMBERT (base), pdf2djvu text | 81.77 (effective_date 85.27, party 78.70, jurisdiction 96.50, term 55.03) | Fine-tuned pipeline, mean of 3 runs (Table 3) | test-A | https://arxiv.org/abs/2105.05796 |
| RoBERTa / BERT / LayoutLM / Flair, pdf2djvu | 81.00 / 78.20 / 78.47 / 77.83 | Same | test-A | https://arxiv.org/abs/2105.05796 |
| LAMBERT on the Tesseract / Textract / Azure CV layers | 81.03 / 77.37 / 80.57 | Table 4, by text layer | test-A | https://arxiv.org/abs/2105.05796 |
| Autotagger (regex, span level) | 60.09 | Table 3 | test-A | https://arxiv.org/abs/2105.05796 |
| Human (annotator agreement, 100 random docs) | 97.86 | Table 3 | test-A | https://arxiv.org/abs/2105.05796 |
| Arctic-TILT (<1B) | 94.3 | Fine-tuned, and Kleister NDA is also in its SFT mix (App. B) (Table 3) | Not stated (UNKNOWN) | https://arxiv.org/abs/2408.04632 |
| ERNIE-Layout (prior SOTA, as cited by Arctic-TILT) | 88.1 | Table 3 | Not stated (UNKNOWN) | https://arxiv.org/abs/2408.04632 |

- This corrects the suite spec, which lists LayoutLMv2-large 85.20 as a test-A number. LayoutLMv2 states: "We report the evaluation results of Kleister-NDA on the validation set because the ground-truth labels and the submission website for the test set are not available".
- The Kleister paper rows are test-A (the table captions say "test sets"). Treat them as reference points, not same-split numbers.
- LayoutLMv2 says only "official evaluation toolkit". Whether its F1 is GEval's upper-cased or true-case variant is UNKNOWN. In our judgement, the gap should be small, because the two differ only by case.

## 7. Contamination risk: high

- The dev-0 gold (`expected.tsv`) and the text have been public on GitHub since 2020-03-05. The paper has been on arXiv since May 2021.
- Arctic-TILT reports Kleister NDA in its SFT data. In our judgement, any LLM from 2024 or later, Clairic or Sage could have seen the repo.
- The survey's L1 check found that Kleister is not among the tasksource extraction-corpus sources.
- Read results as an "old public set" in the suite's old-vs-fresh split (L18).

## 8. Abstention slice (C7) and audit hints

- The 75 null golds feed the pooled business-document null slice: effective_date 21, jurisdiction 5, term 49. Of these, 14 are on keys the official task does not ask for (`meta.null_outside_official_keys`; see §3). Use the 61 officially asked nulls for any abstention claim, or hand-audit the 14.
- `scripts/null_audit_hints.py` is a heuristic and changes no gold. It finds the following:
  - For term, 25 of the 49 null-term documents contain another duration phrase near "term/period/expire/terminate". These are mostly post-employment restriction or survival periods, for example "for a period of two (2) years following cessation of … employment".
    - Kleister's `term` is the contract's own length, so those golds are plausibly right.
    - But the scorer will count an arm that answers with a restriction period as hallucinated.
    - Hand-audit these 25 (30 per source under decision C7) before making any abstention claim.
  - For effective date, 5 of the 21 null-date documents contain a dated reference, mostly to another agreement ("Letter Agreement dated as of January 26, 2001").

## 9. Cost (list-price estimate, `scripts/estimate.out.txt`, from `harness/estimate.py`)

| Arm | Estimate |
|---|---|
| clairic_text | About $0.16 (1.63M input tokens, with 5 fields each re-billing the state) |
| sage_precise | Price not public (0.31M input plus an assumed 54k output) |
| haiku_json | About $0.04 |

These are estimates only. We ran nothing.

## 10. Risks and limits

- n = 83 (duration n = 34), so CIs are wide. The survey planned plus or minus 9.3 pts for a single arm and 10.8 for a paired comparison. Field-level clustering will tighten that somewhat.
- The licence is UNKNOWN (§2).
- Test-A gold is hidden. Every published test-A number is a reference point only, and only LayoutLMv2's rows share our split.
- The COMPARABLE scorer is a re-implementation. Beyond the self-test, we have not verified it against the GEval binary.
- The party-variant rule accepts surface forms with commas. That is lenient relative to Kleister normalisation, and it applies to the TYPED view only.
- Kleister title-cases party values. The TYPED view casefolds and the COMPARABLE view upper-cases, so case never matters.

## 11. Files and rebuild

```sh
U="/opt/homebrew/bin/uv run --no-project --with tiktoken --with httpx python -I"
$U extract-bench/datasets/kleister_nda/build.py                 # items.jsonl, alt/, build_report.json (no network)
$U extract-bench/datasets/kleister_nda/scripts/selfcheck.py     # format + harness-scorer checks -> selfcheck.out.json
$U extract-bench/datasets/kleister_nda/comparable/kleister_geval.py --selftest
$U extract-bench/datasets/kleister_nda/comparable/kleister_geval.py --preds-dir extract-bench/runs/RUN/preds/ARM/kleister_nda
python3 -I extract-bench/datasets/kleister_nda/scripts/null_audit_hints.py
```

- `scripts/explore_gold.py` and `scripts/explore_party.py` are exploration only. `scripts/build.out.txt` is the last build's stdout.
- `raw/repo/` holds the pinned repo files and PDFs.
- `raw/pages/` holds the pages and papers we opened (GitHub API JSON, the README, the three arXiv abstracts and PDFs with their text, and the GEval README and source).
- raw/ is 11.3 MB, and the whole dataset directory is about 16 MB.
