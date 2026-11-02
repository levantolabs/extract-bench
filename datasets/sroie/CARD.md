# sroie: SROIE Task 3 test (ICDAR 2019), as packaged by CC-OCR KIE

Built 2026-10-23 at a cost of $0. No metered API was called. The suite plan is in the suite design notes §2.2 and in decisions L2, L3, L7 and L13.

In brief:
- The set includes all 347 scanned Malaysian receipts and tax invoices.
- The images are byte-identical to the CC-OCR SROIE subset, which produced the published GPT-4o, Gemini, Claude and Qwen rows.
- The gold matches the original SROIE Task 3 labels on 347/347 receipts.
- Each item has two scoring views, and they never share a table:
  - COMPARABLE uses verbatim text and the unmodified CC-OCR scorer.
  - TYPED uses ISO dates and money with 2 decimal places. This view is our adaptation.
- `text` is null, so the harness will OCR the images later.

## 1. Source and revision

| What | Value |
|---|---|
| Dataset file | `kie/constrained_category/sroie2019_word_347.tsv` in https://huggingface.co/datasets/wulipc/CC-OCR |
| HF revision | `c64517e92179991d509776064174776700cdd5a2` (lastModified 2024-12-27) |
| File sha256 | `93b5440238227b46f0eb0a4992a7ba10ece9afd1f09974d18715e91d2122755a` (126,571,413 bytes). It equals the HF LFS oid, and `build.py` checks it |
| TSV columns | index, image (base64 JPEG), image_name, question, answer (gold JSON), category=`kie`, l2-category=`constrained_category`, split=`sroie2019_word` |
| Scorer and prompts | https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Benchmarks/CC-OCR at commit `851672c8005bef8b7cb38c79c51a4fbf9a66d970` (2025-04-09). The local copy is in `raw/cc-ocr-github/`. `kie_evaluator.py` has sha256 `c3f5aa64…8b6b`, and `common.py` has sha256 `1756d8c2…7881` |
| Paper | CC-OCR, arXiv 2412.02210 v3, at https://arxiv.org/html/2412.02210v3 (saved as `raw/pages/ccocr_arxiv_v3.html`) |
| Original dataset | ICDAR 2019 SROIE at https://rrc.cvc.uab.es/?ch=13 and the competition report https://arxiv.org/abs/2103.10213. The Task 3 test set has 347 receipts (LayoutLMv2 Table 1, "SROIE 4 keys, 626/0/347", https://arxiv.org/abs/2012.14740) |
| Cross-check only | jsdnrs/ICDAR2019-SROIE test rows (datasets-server; HF revision `bffe40c26759f3376ec2b3ae9031dbba54cd587c`, GitHub `77c4c11`), saved as `raw/jsdnrs-mirror/test_text.jsonl` (sha256 `e7a38b18…589a`). None of its text goes into the items |

The provenance checks in `build_report.json` found:
- All 347 CC-OCR receipts are in the mirror.
- The gold differs on 0/347 receipts.
- Image width and height are identical on 347/347.
- The mirror has 14 extra test receipts. The mirror author added these annotations ("14 receipt annotations missing from the original Task 3 test dataset", from the mirror README). CC-OCR leaves them out, and so do we.

## 2. Licence

| Layer | Licence | Evidence | Status |
|---|---|---|---|
| CC-OCR packaging and scorer | MIT ("Copyright 2023 Alibaba Cloud") | `raw/cc-ocr/LICENSE` at the HF revision, the HF card (`license: mit`) and GitHub `Benchmarks/CC-OCR/LICENSE` | Verified |
| SROIE images and labels | CC BY 4.0, as the mirror claims | The jsdnrs NOTICE (`raw/jsdnrs-mirror/NOTICE`) and README say "licensed under the CC-BY-4.0 as noted on … https://rrc.cvc.uab.es/?ch=13&com=downloads" | UNKNOWN at the source |

Why the source licence is UNKNOWN:
- On 23 Oct the RRC site timed out under curl and refused the connection under WebFetch (`ECONNREFUSED 158.109.8.55:443`).
- The Wayback copy of the downloads page (https://web.archive.org/web/20260102055801/https://rrc.cvc.uab.es/?ch=13&com=downloads, saved in `raw/pages/`) shows only "You will need to register to get access to the download section". No licence text is visible without registering.
- The competition report (arXiv 2103.10213) has no licence statement.
- For the redistribution status, see `LICENSE-DATA.md`.

## 3. Items and sampling

- The set has n = 347 items. That is the whole CC-OCR SROIE subset, the same set behind CC-OCR Table 5. Items keep their TSV order, and nothing is sub-sampled.
- Seed 20 is used only for the sealed 20% holdout (suite design decision L13).
  - `random.Random(20).sample(sorted(ids), 69)` sets `meta.holdout = true` on 69 items.
  - Holdout items go to Sage only on the final run.
- Each item has one JPEG image at `img/<image_name>`, so the page count is 1.
  - 0 images were re-encoded. The bytes are the decoded TSV base64, unchanged.
  - Image sizes are p50 131 KB, p95 1.09 MB and max 1.82 MB. All are within Clairic's 4 MiB limit.
- Every item has `text: null` and `text_source: "none"`.
  - CC-OCR ships images only. The harness OCR text gets added later.
  - The mirror's Task 1/2 transcripts can't be used, because every line in them is cut at its first comma. 0 of 19,386 test lines contain a comma, while 347/361 gold addresses do.
  - The original RRC transcripts need registration.
- Two prompt strings occur in the TSV, and they differ only in schema key order. 346 receipts use company, date, address, total, and `X51006555819.jpg` uses company, address, total, date. `meta.comparable.prompt` holds the verbatim prompt for each item.

## 4. Fields

| Field | TYPED view (`fields`, metric `exact_norm`) | Gold (TYPED) | COMPARABLE view (`meta.comparable.fields`) |
|---|---|---|---|
| company | text | Verbatim CC-OCR gold | text, verbatim gold |
| date | date | ISO `YYYY-MM-DD`, by the pre-registered rule below | text, verbatim gold (for example `15/01/2019` or `22 MAR 18`) |
| address | text | Verbatim CC-OCR gold | text, verbatim gold |
| total | money | `"%.2f"` string, with the currency (RM/$) stripped | text, verbatim gold (for example `RM 10.60` or `$8.20`) |

- There are 1,388 fields in total (4 per item).
- 0 fields have null gold. SROIE has no abstention items.
- The type map for each arm (suite design decision C2) is:
  - Clairic maps date to `date`, and money to `number` with `decimals: 2`.
  - Sage 2 uses `date` and `money`.
  - The COMPARABLE view uses `text` for all four fields in every arm.

**Pre-registered TYPED conversion, gold only.** We wrote this conversion before any model output existed. The code is `typed_date()` and `typed_money()` in `build.py`.
- The date rule reads dates day-first, as Malaysia writes them. These are the patterns and their counts:

  | Pattern | Count | Example |
  |---|---:|---|
  | `D/M/Y` numeric (`/ - .` separators; YY becomes 20YY) | 296 | |
  | Day, month name and year | 47 | `22 MAR 18`, `27/MAR/2018` |
  | `YYYY-MM-DD` and `YYYY/MM/DD` | 3 | |
  | Invalid when read day-first | 1 | |

  Counting converted dates by year, 2016 has 31, 2017 has 106, 2018 has 207, and 2019 has 3.
- The money rule works like this:
  - It strips `RM`, `MYR`, `$` and spaces.
  - It treats a `,` followed by exactly 3 digits as a thousands separator.
  - It outputs `"%.2f"`.
  - The counts are plain 285, currency stripped 62, and 0 unparsed.
- Six golds are flagged in `meta.typed_gold.notes` and listed for blind review. The gold itself stays unchanged.
  - `X51006414715` has gold `4/22/2018`, which is not a valid date when read day-first. The receipt line is `4/22/2018 08:10PM`, from a US-format POS. Its typed gold is the list `["2018-04-22"]`, the only list gold in the set.
  - `X51005719895` (`1/8/2018`), `X51006414532` (`6/4/2018`) and `X51006619784` (`7/6/2016`) are unpadded and ambiguous. The rule keeps the day-first reading.
  - `X51006556654` (−6.42) and `X51006556723` (−5.09) have negative totals in the gold, possibly from credit notes. They stay as they are.

## 5. Metrics

COMPARABLE is the only score that may sit next to the Table 5 numbers.
- The scorer is the unmodified `raw/cc-ocr-github/evaluation/evaluator/kie_evaluator.py`. It runs with `comparable/label.json`, which holds the verbatim gold JSON for each image_name, and `comparable/qa.jsonl`, which uses the CC-OCR layout with image paths `../img/…`.
- The metric is micro field-level F1 = TP / (TP + (FP+FN)/2).
- Normalisation is minimal.
  - It only converts full-width characters to half-width and cleans up spaces around punctuation and between digits and letters (`kie_evaluator.py` lines 330 to 369).
  - It does not lower-case anything, so one wrong character fails the field.
- The scorer sanity check (`scripts/selfcheck.out.json`) gives these results:

  | Prediction | F1 |
  |---|---|
  | Gold JSON as the prediction | 1.0000 |
  | Empty JSON | 0.0000 |
  | Gold with ISO dates substituted | 0.7522 (date-class F1 0.0086) |

  So a typed date output costs about 25 F1 points on this scorer, and COMPARABLE arms must request `date` as `text` (verbatim).
- In our judgement, the COMPARABLE run needs a caveat on comparability.
  - The VLM and LLM control arms can send the CC-OCR prompt verbatim.
  - Clairic and Sage take typed fields, not a free prompt. Their COMPARABLE run uses the same images, key set, gold and scorer. Their per-field instructions come only from the CC-OCR prompt sentences (`meta.comparable.fields`).
  - So the protocol is "same set, same scorer, prompt adapted", not byte-identical. Say so next to any number.

TYPED carries the label "our adaptation, not comparable".
- Its metric is `exact_norm` on the `fields` gold.
- It scores dates as ISO and money as a value with 2 decimal places.
- Text fields get exact_norm's whitespace and case normalisation, which the harness defines.
  - **Deviation, added by the 23 Oct verifier.** `norm_text` in `harness/score.py` keeps internal punctuation, so TYPED is stricter than the CC-OCR scorer about spacing around commas. CC-OCR's `remove_unnecessary_spaces` rewrites `,X` as `, X`. 61 of the 694 company and address golds have irregular comma spacing, for example `...SELATAN,SG BULOH...` and `7/4,KAWASAN PERINDUSTRIAN TAMPOI,81200`. On those golds, a prediction like `SELATAN, SG BULOH` passes COMPARABLE and fails TYPED. Span-returning arms copy the printed spacing, but VLM arms tend to re-space. Before the run, decide whether to report this as a per-arm caveat or to add an address view that ignores punctuation spacing.
- The date field sets no `dayfirst` key. The harness `parse_date` already defaults to day-first and falls back to month-first only when day-first is invalid. So a verbatim `4/22/2018` prediction still parses to 2018-04-22.
- Never print TYPED scores in the same table as COMPARABLE scores.

Coverage and confidence intervals:
- Both views always cover 4/4 fields, because there are no list fields and no booleans.
- The planning half-width is ±4.6 points for a single arm and ±5.3 points paired (suite design notes §2). Use a document-clustered bootstrap with seed 20.

## 6. Published baselines (reference rows, with only the numbers I saw on the opened page)

| System | Score | Setting | Split / n | Provenance | URL |
|---|---:|---|---|---|---|
| Qwen2-VL-72B | 75.50 | Zero-shot, image, CC-OCR prompt, field F1 (CC-OCR scorer) | SROIE test via CC-OCR, 347 | Alibaba/Qwen authors (method-author-run) | https://arxiv.org/html/2412.02210v3 (Table 5) |
| Gemini-1.5-Pro-002 | 75.00 | Same | Same | Same | Same |
| GPT-4o-2024-08-06 | 72.55 | Same. Replication anchor (suite design decision L3) | Same | Same | Same |
| Claude-3.5-Sonnet-20241022 | 69.83 | Same | Same | Same | Same |
| InternVL2-76B | 57.06 | Same | Same | Same | Same |
| GPT-4+OCR | 90.6 | Zero-shot, OCR text only (no spatial information), KIE F1. The baseline's prompt wording is UNKNOWN. DocLLM's own KIE template is per-key ("What is the value for the {key}?") | "unseen test split" (SDDS); n not stated (UNKNOWN) | JPMorgan (DocLLM authors) | https://arxiv.org/abs/2401.00908 (Table 5) |
| LayoutLMv2-LARGE (fine-tuned) | 97.81 | Supervised, official OCR, entity F1 "excluding OCR mismatch", official eval site | Task 3 test 347 | Microsoft (method authors) | https://arxiv.org/abs/2012.14740 (Tables 2 and 8) |
| SROIE RRC top-1 (to 2020-12-24) | 97.67 | Leaderboard, as quoted by LayoutLMv2 | Task 3 test | Quoted secondhand | https://arxiv.org/abs/2012.14740 (Table 2) |

Notes on these rows:
- The API versions are the ones the paper text states. The commercial APIs are GPT-4o-2024-08-06, Gemini-1.5-Pro-002 and Claude-3.5-Sonnet-20241022. The README dates the tests to about 20 Nov 2024.
- The CC-OCR doc `assets/doc/VLMEvalKit_en.md` shows "QwenVLMax … sroie2019_word 76.37" as an example output, not a published result.
- The rows are not interchangeable.
  - The CC-OCR rows are the only like-for-like rows for the COMPARABLE score.
  - The DocLLM text row uses a different prompt and an unspecified F1 implementation.
  - The fine-tuned rows are supervised ceilings.
- The CC-OCR paper (§A.4) notes a known failure mode. GPT-4o tends to standardise dates, for example turning "26 May 2018" into "2018-05-26", and the scorer counts these as errors. That matches the sanity check above.

## 7. Gold noise and adjudication (suite design decision L7)

- **Pre-registered rule.** After the arms run, items where all arms agree with each other but not with the gold get a blind review against the image. The review applies to all arms equally. We report raw and adjudicated scores side by side and never edit the raw gold in `items.jsonl`. The errata file stays local, and publishing it is the maintainers' call.
- A $0 pre-scan (`scripts/gold_noise_scan.py`, output in `scripts/gold_noise_candidates.jsonl`) found 26 candidate items.
  - 24 are tokens that differ from the mirror's human OCR transcript by exactly one character.
  - 12 are company or date gold strings that are not substrings of the transcript.
  - These candidates show where to look. They are not verdicts. Several are transcript errors rather than gold errors, for example OCR `TAMPOL`/`LUMPER` against gold `TAMPOI`/`LUMPUR`.
- I confirmed one item by eye. `X00016469670` prints "81750 MASAI, JOHOR", but the gold address has "B1750". These likely gold typos are still unreviewed: `X51005724623` (gold "CHJAN" against OCR "CHUAN"), `X51007846397` ("INTERANTIONAL"), `X51007846358` ("JAHOR") and `X51005568887` (date 26/04 against OCR 06/04).

## 8. Contamination and other risks

- Contamination is high.
  - The test gold has been public since 2019 (RRC, GitHub and HF mirrors) and in CC-OCR/VLMEvalKit since Dec 2024.
  - DocLLM's instruction tuning used SROIE.
  - Any extraction model's training mix plausibly includes it.
  - Treat wins here with scepticism. The fresher sets in the suite (RealDoc 2026, emails, FATURA) are the check.
- Receipts are over-represented relative to invoices. We accept this because this set is the comparability anchor.
- The content has some quirks.
  - Some receipts carry a handwritten or overlaid name, for example "tan chay yee" on `X00016469670`.
  - The CC-OCR prompt says the images "do not contain any real personal privacy data". That is their claim, and we did not check it.
- Address is the hard field.
  - The gold joins multi-line addresses into one line and keeps the commas printed on the receipt.
  - Span-returning models (Clairic text spans, Sage verbatim) must span across line breaks, and the COMPARABLE scorer is case-sensitive.
  - `gold_reachable` records whether the gold is a span of the harness OCR. It stays `null` until the harness OCR exists.

## 9. Size and tokens (o200k_base, `scripts/selfcheck.out.json`)

| Quantity | p50 | p95 | Max |
|---|---:|---:|---:|
| `text` (null for all items) | 0 | 0 | 0 |
| Proxy document text: mirror Task 1/2 transcript lines joined by `\n` (a lower bound, since lines are cut at commas) | 275 | 446 | 605 |
| CC-OCR prompt | 183 | 183 | |
| Gold JSON | 71 | 87 | |

- Image tokens for Clairic vision are UNKNOWN until the C1 probe runs.
- The verifier added a pixel-size check. 52/347 images are longer than 4096 px on the long side, and 113 are longer than 2048 px. The largest is `X51006414715` at 4961×7016 (34.8 MP, still under 4 MiB). The local Clairic docs state only a byte limit, so any pixel or resolution limit is UNKNOWN. Downscaling would break the byte-identity with CC-OCR. If the C1 probe shows a limit, record any harness-side resize for each arm.
- On disk, `raw/` takes 134 MB and `img/` takes 91 MB. The whole directory is about 228 MB.

## 10. Files and how to rebuild

```
raw/cc-ocr/             sroie2019_word_347.tsv, LICENSE, README.md   (HF rev c64517e9)
raw/cc-ocr-github/      CC-OCR README, docs, index/kie.json, evaluation/ (commit 851672c8)
raw/jsdnrs-mirror/      test_text.jsonl (cross-check only), NOTICE, LICENSE, README.md, github_README.md
raw/pages/              CC-OCR arXiv v3 html/txt, DocLLM pdf/txt, LayoutLMv2 pdf/txt,
                        SROIE report pdf/txt, Wayback RRC pages
img/                    347 JPEGs (byte-identical to the TSV)
items.jsonl             347 items (sha256 c4109b8c…49ff2 at build time)
comparable/             label.json + qa.jsonl in CC-OCR layout for the unmodified scorer
build.py, build_report.json
scripts/selfcheck.py (+ .out.json), scripts/gold_noise_scan.py (+ candidates), scripts/build.out.txt
```

```
/opt/homebrew/bin/uv run --no-project --with pillow==11.3.0 python -I datasets/sroie/build.py
/opt/homebrew/bin/uv run --no-project --with pillow==11.3.0 --with tiktoken==0.12.0 --with zss==1.2.0 \
    --with nltk==3.9.1 --with tabulate==0.9.0 python -I datasets/sroie/scripts/selfcheck.py
/opt/homebrew/bin/uv run --no-project python -I datasets/sroie/scripts/gold_noise_scan.py
```

- `selfcheck.py` loads only `kie_evaluator.py` and `common.py` from the CC-OCR repo. I read both in full first. It does not execute the package `__init__`, which imports the other evaluators.
- We ran no dataset loading script.
