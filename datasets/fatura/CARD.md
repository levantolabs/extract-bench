# fatura: FATURA invoices, inter-template held-out split (Strat2) [synthetic]

This set has 150 single-page invoice images from the 10 templates that FATURA's own Strat2 split holds out, 15 per template, drawn with seed 20. Each invoice has 11 typed fields. Of those, 1,470 field slots are scored: 1,095 have a value and 375 have a null gold. A null gold marks a field we verified as absent from the page. Of the null slots, 150 are `po_number`, which no held-out template prints.

The data is synthetic. Totals never reconcile (0/90 items), and the due date falls before the invoice date on 54/105 items, which is every item that prints both dates. So the set tests reading values off an invoice layout the system has not seen. It does not test reasoning about real invoices.

Only one zero-shot VLM field-accuracy figure has been published on FATURA. In arXiv 2609.20110, Qwen3.6-27B gets 60.2% and Gemini-3.1-Flash-Lite gets 63.9% of fields correct. That paper uses its own 200-doc sample and a 14-field taxonomy, so its figure is a reference only and is not comparable.

## Source and version
| Item | Value |
|---|---|
| Dataset | FATURA Dataset by Limam, Dhiaf and Kessentini (Digital Research Center of Sfax) |
| Record used | https://zenodo.org/records/10371464 (v2, published 2023-12-13; API `revision` 4). It is the latest of the 2 versions listed at `https://zenodo.org/api/records/10371464/versions` |
| File | `FATURA2.zip`, 690,717,658 bytes, `md5:4c9404462f22c5241eb1a290a02eb2a2`. We did not download it in full. `build.py --fetch` range-reads members and checks each one's CRC32 against the zip central directory |
| v1 (not used) | https://zenodo.org/records/8261508 (2023-08-06, `FATURA.zip` 363.5 MB). The paper links v1. v2 adds the 10,000 colored-background images |
| Paper | https://arxiv.org/abs/2311.11856 (v1, 20 Nov 2023). The PDF and text are saved as `raw/arxiv_2311.11856v1.{pdf,txt}` |
| Metadata snapshot | `raw/zenodo_api_record_10371464.json` (fetched 2026-10-23) |
| Contents (record page) | 10,000 white-background JPGs, 10,000 colored-background JPGs and 3 sets of 10,000 JSON annotations (Original, COCO, HF/LayoutLMv3). There are 50 templates with 200 images each, and 24 classes |

## License
- The data is licensed CC BY 4.0. The record page https://zenodo.org/records/10371464, shows "Creative Commons Attribution 4.0 International", and the API gives `license.id = cc-by-4.0` (`raw/zenodo_api_record_10371464.json`). v1 has the same license. Attribution is required, so cite Limam et al. 2023 or the Zenodo DOI 10.5281/zenodo.10371464.
- The arXiv abs page lists the paper under CC BY-NC-SA 4.0. That license covers the paper text only, not the data.

## What was downloaded (`raw/`, 64 MB against a 1.5 GB cap)
The directory holds the following files:
- the split files `Strat2_Split.txt` and `strat1_{train,dev,test}.csv`;
- `Annotations/Original_Format/*.json`, all 10,000, which we used to scan the held-out population;
- `Annotations/layoutlm_HF_format/*.json` for the 2,000 held-out-template files, kept for provenance but not used by the build;
- `images/Template{t}_Instance{k}.jpg` for the 150 sampled white-background images only;
- `zip_index.json` (the central directory, including offsets) and `fetch_manifest.json` (sha256 of the split files and images).

The fetch read the annotation block in one 25.2 MB range read and each image in its own range read.

## Sampling
- The pool comes from Strat2, the inter-template split. `Strat2_Split.txt` gives `dev_inds = [50, 7, 32, 39, 2, 12, 4, 49, 10, 25]` and `test_inds = dev_inds`, and the other 40 templates are train. The file lists template ids only, with no image-level dev/test partition, so all 200 instances of each held-out template are eligible. That gives a pool of 2,000. The Zenodo description says the Strat2 validation and test sets are "the same templates but of different images". That image partition is not shipped, so our 150 may mix both.
- The draw uses `random.Random(20)`. For each template, in the file's order, it calls `sample(range(200), 15)` and sorts the result. That gives n = 150 (15 per template across 10 templates). `build.py` is deterministic: two runs gave the same `items.jsonl` sha256, `e2f9edbf…90b1f`.
- The images are the white-background `images/` files, copied byte-for-byte. None was over 4 MiB, so none was re-encoded. Every image is 595 by 841 px (A4 at 72 dpi) and 1 page. The colored-background variant has the same stem under `colored_images/`. We did not download it.

## Field mapping (our own label-stripping adapter, declared)
Each FATURA region's `text` includes its printed label, for example `"Invoice Date: 14-Apr-2022"`, `"SUB_TOTAL : 885.25  USD"` or `"INVOICE ID INV52959005"`. The adapter strips the label. We first enumerated every label/value pattern in the 2,000 held-out docs (`tools/labels_scan.out.txt`). The adapter then ran on all 2,000 with 0 failures (`tools/selfcheck.out.json`).

| Field | Type | Source class | Gold form |
|---|---|---|---|
| `invoice_number` | text | NUMBER, with `INVOICE #`, `INVOICE ID` or `Invoice number` stripped | Verbatim, e.g. `INV/78-64/532`, `3Y5M9d-848` |
| `invoice_date` | date | DATE (`Date:` or `Invoice Date:`) | ISO `YYYY-MM-DD`. The page prints `DD-Mon-YYYY`, which is unambiguous |
| `due_date` | date | DUE_DATE | ISO |
| `po_number` | text | PO_NUMBER | Null on all 150 items, because no held-out template has the class and none of the audited pages prints a PO |
| `seller_name` | text | SELLER_NAME | Verbatim |
| `buyer_name` | text | BUYER (`Buyer :` or `Bill to:`) or BILL_TO (`BILL_TO:`), first line, with the label stripped | T49 prints both a "Buyer" block and a "BILL_TO" block with different names, so either name is accepted (list gold) |
| `subtotal` | money | SUB_TOTAL | Number, 2 dp. The currency (USD, EUR or $) is kept in `meta.gold_source` |
| `tax` | money | TAX (`TAX:VAT (r%): x`), or a single `GST(n%)` class (T2, T7) | Number. Only the amount is used, and the rate is dropped |
| `discount` | money | DISCOUNT (`DISCOUNT(r%): (-) x`) | `[x, -x]`, because the sign convention is not part of the test |
| `total` | money | TOTAL (on T49 this region reads `BALANCE DUE : x`) | Number |
| `amount_due` | money | AMOUNT_DUE (absent in all held-out templates), or TOTAL when it is labelled `BALANCE DUE` | T49 gets the value. T39 (no total of any kind) gets null. Elsewhere the field is excluded (see below) |

The field `instructions` describe the field only and give no hint about absence. Each system abstains with its own native null.

## Gold and null audit (hand check of page images)
We viewed 15 page images: at least one per template, second instances of T10, T32 and T39, and two OCR-miss cases. A class absent from the annotation gives a null gold only if nothing on the page answers the field. Otherwise we exclude the field from that item and record the reason in `meta.excluded_fields`.

| template | invoice_number | invoice_date | due_date | po_number | seller_name | buyer_name | subtotal | tax | discount | total | amount_due |
|---|---|---|---|---|---|---|---|---|---|---|---|
| T50 | null | val | val | null | val | val | val | val | alt | val | EXCL |
| T7 | val | val | val | null | val | val | val | val | null | val | EXCL |
| T32 | val | val | null | null | val | val | val | EXCL | null | val | EXCL |
| T39 | val | val | val | null | EXCL | val | val | val | alt | null | null |
| T2 | val | val | null | null | val | val | val | val | null | val | EXCL |
| T12 | val | val | val | null | val | val | val | val | alt | val | EXCL |
| T4 | null | val | val | null | val | val | val | null | alt | val | EXCL |
| T49 | val | val | null | null | val | alt | val | null | null | val | val |
| T10 | val | val | val | null | val | val | null | EXCL | alt | val | EXCL |
| T25 | val | val | val | null | val | val | val | EXCL | null | val | EXCL |

In the table, `alt` means a list gold: the discount accepts x or -x, and the T49 buyer accepts either name. `tools/field_tokens.py` generated the table.

We exclude these fields from scoring:
- `tax` on T10 and T32. The line-item table has a TAX column showing `$0.00`, and there is no tax total, so "0.00" versus null is ambiguous. We saw this on T10 I11/I191 and T32 I103/I160.
- `tax` on T25. The page has five separate `GST(n%)` lines and no single tax amount.
- `seller_name` on T39. There is no seller-name text, but the template logo has the letters "LCSR", which a reader may fairly take as the seller.
- `amount_due` on 8 templates (120 items). This deviates from the spec's field plan, which had a null gold here. No "amount/balance due" label is printed, but a TOTAL is. In our judgement, the total is the amount due in AP practice, so a null gold would punish a reasonable answer.

These null slots stay in the set (375 in all, or 225 excluding `po_number`):
- `invoice_number` on T50 and T4, which print no number anywhere. T50 has a "Bank Account Number" distractor.
- `due_date` on T32, T2 and T49.
- `subtotal` on T10, which prints only the discount and total.
- `tax` on T4 and T49, which print no tax amount. T4 I0 is titled "TAX INVOICE", and T49 shows only a GSTIN.
- `discount` on T7, T32, T2, T25 and T49.
- `total` and `amount_due` on T39, which prints the subtotal, tax and discount but no total.
- `po_number` on all 150 items.

The null golds are template-level. About 10 templates give at most 10 distinct "absence patterns", so the effective n for abstention is far below 375. Report abstention per template.

We checked that the annotations agree with the images:
- Tesseract 5.5.0 (local, free, psm 3, eng) finds 1,053/1,095 non-null golds in the text of 2x-upscaled images, and 982/1,095 at native 72 dpi (`tools/selfcheck.out.json`). We ran it as a sanity check only. It is not the harness OCR.
- `tools/misses.py` lists the 42 misses at 2x. We spot-checked 2 of them on the page images (T50 I19 subtotal 1005.69 and T12 I29 tax 43.29), and both are printed exactly as annotated. For the rest, the nearest OCR lines look like OCR errors, not gold errors. For example, `30-Jul-2001` reads as `30-jul-2004`, `4Y4M5d` reads as `4y4msd`, and the text next to the logo on T49 is garbled. We did not view the other 40 one by one.
- 72 dpi is low for OCR, so the choice of harness OCR engine will move Sage-text scores.

We do not use the annotation's `OTHER` class. It holds a full-page text, but that text does not match the v2 images. For example, T49 `OTHER` begins "Shri Ganesh Catering Services", while the page shows "Robinson-Carson". So `text = null` and `text_source = "none"`, and the harness OCRs the image.

## Metric
- The metric is `exact_norm` per field, after type normalisation:
  - dates become ISO;
  - money becomes a number (2 dp; the currency is not scored);
  - text is case- and whitespace-normalised;
  - for a list gold, any listed value is correct.
- We chose it because invoice fields are short, typed and unambiguous once normalised. AP use needs the exact value, so partial credit (ANLS) would flatter the scores.
- Score null-gold slots (false-positive extraction) separately from value slots, using the suite's abstention rules (a) and (b).
- The metric is not comparable to the paper's metrics (below).

## Published baselines (reference only; none is comparable)
The rows from the dataset paper (2311.11856) use models fine-tuned on FATURA's 40 training templates. They score region-level classification (detection mAP, or F1 over region or token labels), not field-value exact match.

The verifier made a correction on 2026-10-23. The builder had written that "no zero-shot or LLM row exists", and that was wrong. arXiv 2609.20110 reports zero-shot VLM field-value correctness on FATURA (rows below). A FLLM2024 paper also runs Llama3 and Mistral over Tesseract and DocTR OCR on FATURA, but its numbers are UNKNOWN because the PDF is behind a bot wall.

| System | Score | Setting | URL |
|---|---|---|---|
| LayoutLMv3 "reference approach" (GT region boxes and GT text at inference; the authors call it an upper bound that is "not practically viable") | F1 95.7%, R 97.5%, P 95.7% | Strat2 inter-template test (Table 5) | https://arxiv.org/abs/2311.11856 |
| LayoutLMv3 reference approach | "99% across all metrics" (in the text, no table) | Strat1 intra-template val/test | https://arxiv.org/abs/2311.11856 |
| Hybrid: YOLOS regions, then EasyOCR, then LayoutLMv3 token labels | F1 56.3%, R 47.7%, P 70.7% | Strat2 test (Table 7) | https://arxiv.org/abs/2311.11856 |
| Hybrid | F1 88%, R 84%, P 92% | Strat1 test (Table 6) | https://arxiv.org/abs/2311.11856 |
| YOLOS (region detection only) | mAP@IoU50 43.58%, mAP@IoU75 32.7% | Strat2 test (Table 4) | https://arxiv.org/abs/2311.11856 |
| YOLOS | mAP@IoU50 97.57%, mAP@IoU75 81.84% | Strat1 test (Table 3) | https://arxiv.org/abs/2311.11856 |
| Qwen3.6-27B, zero-shot, one ungrounded prompt over the page image (greedy) | 60.2% of 1,850 field extractions correct (canonical match: ISO dates, half-cent money tolerance, normalised strings) | The paper's own 200-doc FATURA eval sample (split and templates not stated; layout index from the training split), with 14 field types mapped from the 24 classes. Table 4, Appendix C. This is the closest to our task but still not comparable: the sample and taxonomy differ, and the denominator counts extracted fields | https://arxiv.org/abs/2609.20110 |
| Gemini-3.1-Flash-Lite, same setting | 63.9% of 1,767 field extractions correct | Same as above | https://arxiv.org/abs/2609.20110 |
| Llama3 and Mistral over Tesseract and DocTR OCR ("LLM-centric pipeline for information extraction from invoices", FLLM2024) | UNKNOWN (an Anubis bot wall blocks the PDF; we read only the HAL abstract) | SROIE and FATURA; setting UNKNOWN | https://hal.science/hal-04772570 |

We read the 2311.11856 numbers from the PDF text in `raw/arxiv_2311.11856v1.txt`, and the verifier checked them against a fresh arXiv download with the same PDF sha256. The verifier read the 2609.20110 numbers from https://arxiv.org/pdf/2609.20110 (submitted 17 Sep 2026).

The 56.3% Strat2 hybrid row is the only fine-tuned "realistic pipeline on unseen layouts" figure. It shows that inter-template generalisation is hard for fine-tuned layout models, but it is still a different task from ours. The zero-shot VLM rows at about 60-64% suggest that near-frontier VLMs are far from ceiling on FATURA field values. In our judgement, that gap is partly an artefact of the paper's taxonomy and canonicaliser, and it stays UNKNOWN until we run our own.

## Contamination risk: low to medium (our judgement)
- The images and full gold have been public on Zenodo since Aug 2023 (v1) and Dec 2023 (v2). They sit inside a zip, not in crawlable HTML.
- The values are Faker-style random text, so memorising them gains little. Memorising the 50 layouts could help.
- FATURA is not among the 127 tasksource extraction-corpus train sources (suite design notes, L1).
- Whether Sage 2 or Clairic trained on FATURA is UNKNOWN.

## Token and size estimates
- The OCR text from Tesseract has these o200k token counts per item:
  - p50 219 and p95 331 at native resolution;
  - p50 233 and p95 317 at 2x (range 124 to 367).
- The field definitions (11 fields as JSON) take p50 260 o200k tokens.
- Sage's 16,384-token cap is not a problem.
- Image tokens for Clairic vision are UNKNOWN. Each image is a 595 by 841 JPEG of about 40 KB (max 65,883 B in the zip).

## How text was produced
`text` is null in `items.jsonl` (`text_source: "none"`). The harness supplies OCR from its one free local engine. The Tesseract text from the sanity pass is in `tools/tesseract_txt/`, at 1x and 2x scale. It is not the harness text, and no item references it.

## Risks and limitations (blunt)
- The data is synthetic, and templates generate the values:
  - the total matches subtotal plus tax minus discount on 0/90 checkable items (`tools/consistency.out.json`);
  - the due date is earlier than the invoice date on 54/105 items, which is all items with both dates. An earlier count of 45/90 came from a bug in `tools/consistency.py` that skipped the date check on tax-excluded items. The verifier fixed it on 2026-10-23.
  - A system that sanity-checks or recomputes amounts is penalised, and one that copies blindly is not.
  - The subtotal does equal the sum of the line amounts (seen on T50 I19), as the paper states.
- There are only 10 layouts. Template variance dominates, so report results per template and use template-clustered bootstrap CIs.
- These are not real invoices. Evidence on real invoices needs DocILE (signup required) or a holdout we build.
- We wrote the adapter and chose the exclusions, including one deviation from the spec (amount_due). All of them are listed above and coded in `build.py` (`EXCLUDE`, `gold_for`).
- The images are low resolution (72 dpi). That makes the OCR engine a confound for Sage, which reads text only.

## Files
- `build.py` has two modes. With `--fetch` it range-reads from Zenodo into `raw/`. By default it builds `items.jsonl` and `img/` from `raw/` only. Run it with `/opt/homebrew/bin/uv run --no-project --with remotezip|pillow python -I build.py`.
- `tools/` holds these scripts and outputs:
  - `selfcheck.py` and `.out.json`, which run the format checks, the adapter over 2,000 docs, Tesseract reachability and token counts;
  - `labels_scan.py` and `.out.txt`;
  - `explore.py` and `.out.txt`, which show class presence per template;
  - `misses.py`;
  - `consistency.py` and `.out.json`;
  - `field_tokens.py` and `.out.json`.
