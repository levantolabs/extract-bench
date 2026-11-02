# cord: CORD v2 receipt photos (test 100 + dev 100)

`build.py` built this set on 2026-10-23. The build runs offline, costs $0, and calls no metered API. It is also deterministic. Two runs gave the same `items.jsonl` sha256, `736c73ee…0d86`, after the verifier pass of 2026-10-23. Before that pass added `audit/verifier_gold_audit-2026-10-23.json`, the sha256 was `8a22259a…288c`.

This is a Tier A, existing-adapted set built on an independent dataset from NAVER Clova. We built only the typed-field adapter and the hand audit. The receipts, the gold and the comparable prompts all come from the source.

## Source
| | |
|---|---|
| Dataset | CORD v2 (Consolidated Receipt Dataset for post-OCR parsing), a set of Indonesian shop and restaurant receipt photos |
| URL | https://huggingface.co/datasets/naver-clova-ix/cord-v2 (homepage https://github.com/clovaai/cord) |
| Revision | HF commit `7f0115a4b758a71d6473b8d085751692da2fef98` (lastModified 2022-07-19T23:43:33Z). GitHub `clovaai/cord` master is at `327310ce58c1623255821d062b3a759ff3789e3c` |
| Files used | `data/test-00000-of-00001-9c204eb3f4e11791.parquet`: 234,202,795 B, LFS sha256 `51c65f1788faff392abe2a0b55b023eb23e9be551c509138eaa3a832514224e7`<br>`data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet`: 242,080,800 B, sha256 `0d0f6dac11fdcc549de2746aa9f53136a3bc22a2a1aff2b0b847f7622ad60c15`<br>We verified both sha256 values after download, and `build.py` refuses to run if either hash does not match |
| License | CC BY 4.0, verified in four places:<br>• the HF card says `license: cc-by-4.0` (`raw/hf_README.md`)<br>• HF `dataset_infos.json` says "Creative Commons Attribution 4.0 International License"<br>• the GitHub README says "This work is licensed under a Creative Commons Attribution 4.0 International License", and the repo has a `LICENSE-CC-BY` file (`raw/github_cord_README.md`, `raw/github_cord_LICENSE-CC-BY.txt`)<br>• the GitHub API license field is `CC-BY-4.0` (`raw/github_cord_repo.json`)<br>The licence text is at http://creativecommons.org/licenses/by/4.0/. Attribution is required, so cite Park et al. 2019 |
| Size | 1,000 receipts: 800 train, 100 dev and 100 test (README download table). The raw download here is 485 MB (the test and validation parquet files, the CC-OCR TSV and metadata). We did not fetch train. The whole dataset directory is 754 MB |
| COMPARABLE source | The CC-OCR KIE track, https://huggingface.co/datasets/wulipc/CC-OCR at revision `c64517e92179991d509776064174776700cdd5a2`. The card licence is MIT, but CC BY 4.0 still governs the CORD images.<br>The file is `kie/constrained_category/CORD_100.tsv` (20,165,555 B, sha256 `814b0d52…807d`).<br>The scorer is CC-OCR `evaluation/evaluator/kie_evaluator.py` at `AlibabaResearch/AdvancedLiterateMachinery@851672c8005bef8b7cb38c79c51a4fbf9a66d970`, copied unmodified to `raw/cc-ocr-github/` |

The README says some classes were removed: "Some class labels shown in the original paper were removed due to Indonesian legal issues. In particular, the `store_info`, `payment_info`, and `etc` fields have been removed" (https://github.com/clovaai/cord). All our fields are in `menu`, `sub_total` and `total`, so the removal does not affect them. The images themselves carry privacy blur over store headers and some payment lines.

## Sampling
- Setting `sample_n` = 200 takes every receipt in test (100) and dev/validation (100). Nothing is subsampled. The CORD `meta.split` value for dev is "valid".
- Seed 20 only marks the sealed 20% holdout (suite design decision L13), stratified by split. In all, 20 test items and 20 dev items have `meta.holdout` = true.
- The CC-OCR-comparable score uses all 100 test items, including the holdout. We add dev for statistical power (suite design decision C13).
- Item ids take the form `cord-test-NNN` or `cord-dev-NNN`, where NNN is the CORD `meta.image_id`.
- The build found near-duplicates. In 5 pairs, both receipts share the same menu multiset and the same typed golds.
  - The pairs are dev-014/test-084, dev-016/dev-038, dev-027/dev-044, dev-059/test-038, and dev-073/test-068.
  - 3 of the pairs have an identical `gt_parse`. Two of those (dev-016/dev-038 and dev-059/test-038) look like the same photo framing. We checked the images visually.
  - Every pair involves a dev item. 3 pairs cross test and dev, and 2 sit inside dev. Dev therefore adds less independent power than 100 receipts would.
  - The build sets `meta.dup_group` and `meta.cluster_id`, so bootstrap by `cluster_id`. That leaves 195 effective receipts.

## Item layout (common item format)
- Each item has `modality` = image, and `images` holds one receipt photo.
  - All 200 sources are PNG. The build copies 172 of them byte-for-byte.
  - 28 PNGs were over 4 MiB (4.5 to 17.2 MB). The build re-encoded them to JPEG at quality 92 and kept the pixel size. The largest output is 2.16 MB.
  - Sizes run from 228×336 to 4,224 px on the long side. No image has an EXIF orientation tag, and `page_count` = 1.
- Items carry no text, so `text` = null and `text_source` = "none".
  - CORD ships no OCR-engine output, only human word transcripts.
  - The suite's primary text for Sage 2 and Clairic-text is the harness OCR in reading order (suite design decision L6), which the harness produces later.
- The gold-word oracle arm (`meta.oracle_text`) is labelled as an oracle and is never called an "OCR ceiling".
  - The build makes it from the `valid_line` human word transcripts. It groups words into lines by the annotators' `row_id`, orders the lines by median quad y-centre, and orders the words within a line by quad x-min.
  - The build uses `row_id` instead of pure y-clustering because that grouping keeps the lines of tilted photos intact. On test, 785/791 rows span less than one line height. The other 6 are tilted photos plus one row that joins TOTAL and CASH.
  - The text has no distractors. CORD transcribes only labelled words, so the store, address, dates and payment details are missing.
  - This deviates from the spec, which said "plus dontcare". CORD v2 `dontcare` entries are polygons with no text (we checked, and they hold quad lists only), so there was nothing to add. 21/200 items have at least one dontcare region.
- The item-level metric is `metric` = exact_norm. The list fields use list_f1 (`meta.field_metrics`).

### TYPED fields ("our adaptation, not comparable to CC-OCR/LMDX")
| field | CORD key | type | gold present | gold null | dropped |
|---|---|---|---:|---:|---:|
| subtotal_price | sub_total.subtotal_price | money | 132 | 68 | 0 |
| discount_price | sub_total.discount_price | money | 11 | 188 | 1 |
| service_price | sub_total.service_price | money | 25 | 175 | 0 |
| tax_price | sub_total.tax_price | money | 88 | 110 | 2 |
| total_price | total.total_price | money | 193 | 7 | 0 |
| cashprice | total.cashprice | money | 131 | 69 | 0 |
| changeprice | total.changeprice | money | 120 | 80 | 0 |
| creditcardprice | total.creditcardprice | money | 30 | 170 | 0 |
| menuqty_cnt | total.menuqty_cnt | number | 55 | 145 | 0 |
| menu_nm | menu.nm | list (text) | 200 | 0 | 0 |
| menu_cnt | menu.cnt | list (number) | 181 | 19 | 0 |
| menu_price | menu.price | list (money) | 198 | 2 | 0 |

- There are 2,397 fields in total.
  - Of the 1,797 scalar fields, 785 have gold and 1,012 are null (56.3%).
  - Of the 600 list fields, 579 have gold and 21 are null. They hold 1,368 gold list entries.
- **Read this first.** A model that always answers null scores 56.3% on scalar exact_norm.
  - Never report a pooled scalar score alone. Report three numbers: present-gold accuracy, null-gold accuracy (abstention) and the pooled score.
  - The null slots feed the pooled null-gold business-doc slice (suite design decision C7).
- Null gold means the key is absent from `gt_parse`. For example, 34/100 test receipts and 33/100 dev receipts have no `sub_total` block at all. `gt_parse` holds only value words (`is_key`=0).
- The hand audit of null golds (suite design decision C7) is in `audit/null_gold_audit-2026-10-23.json`, written by `scripts/null_gold_audit.py`.
  - The sample is 19 receipts with 107 null slots. It covers all 7 receipts with a null `total_price`, plus 12 drawn with seed 20.
  - In the random stratum, the audit confirmed 60/60 null slots (56 absent, 4 privacy-masked).
  - In the total_price-null stratum (47 slots), 42 are absent, 1 is masked, 1 is a wrong null, 2 are ambiguous and 1 is unclear.
    - The wrong null is on cord-test-037. The receipt prints "TOTAL 20.000", but CORD files it under `total.total_etc`.
    - Two slots are ambiguous. cord-test-009 prints "Discount (0 %)" with no amount, and cord-test-076 prints "ITEMS 5", which CORD labels `menutype_cnt`.
    - One slot is unclear. On cord-test-039 a TOTAL line is cut off at the photo edge.
  - `fields[].gold` stays the raw CORD gold, and wrong golds go in `meta.gold_adjudicated`. Report raw and adjudicated scores side by side (suite design decision L7).
- An independent agent ran a verifier pass on 2026-10-23, and its verdicts are judgement calls. They are in `audit/verifier_gold_audit-2026-10-23.json` and merged as `meta.null_gold_audit.verifier_audit`.
  - The pass was targeted, not random. It looked at null slots whose field keyword appears in the oracle text (18/1,012 hits, a lower bound) and at receipts that fail the check cash − total = change. The verifier checked every hit on the full-resolution image. These counts are not an error rate.
  - It added 3 wrong golds to `meta.gold_adjudicated`:
    - cord-test-095 changeprice = 201. The receipt prints "KEMBALI 201", and CORD files it as a second `cashprice` value.
    - cord-test-095 cashprice = 62000 (`wrong_value`). The raw lenient alternatives [62000, 201] include that mislabelled change amount.
    - cord-test-071 subtotal_price = 198000. The receipt prints "Sub Total 198000", and CORD files it as `total_price`.
  - It recorded these ambiguous slots without adjudicating them:
    - cord-test-009 total_price (36000) and cord-dev-029 total_price (30000). These receipts print only "Sub Total", followed by cash and change. The first audit had marked these slots `absent`.
    - cord-dev-021 discount_price (34363). This is a whole-bill "Discount BCA 15%" line that CORD files as a menu item, so it also sits in the `menu_nm`/`menu_price` list gold.
  - CORD labelling is inconsistent. CORD files the same "Sub Total"-only layout as `subtotal_price` on test-009/dev-029 but as `total_price` on test-071. Item-level discount lines (for example test-031/033 "Disc -100% (ITM06)") have null `discount_price`. The field instruction says "whole bill", so they stay null.
  - The verifier was Claude, working in a single pass on about 500 px previews and not blind to the gold. Its verdicts are judgement calls, not human adjudication.
  - `meta.null_gold_audit.needs_review` also flags 28 items where a null scalar sits next to an unasked key in the same section (`sub_total.etc`, `othersvc_price`, `total_etc`, `emoneyprice` or `menutype_cnt`).
- Gold strings go through a pre-registered conversion, the "Indonesian thousands rule" (suite design decision L2). The code is in `typed_money()` and `typed_count()`. The rules fired as follows: money thousands 645, plain 70, decimal mark 20, and unparsed 3.
  - For money:
    - Strip `Rp`/`Rp.`/`IDR`, spaces (so "154 000" becomes 154000) and the characters `* [ ] : =`.
    - `(x)` or a leading `-` means negative.
    - A last group of 1 or 2 digits is a decimal part. Otherwise every `.` or `,` followed by exactly 3 digits is a thousands separator.
    - For example, "5.455" becomes 5455, "9.500,00" becomes 9500, and "48982.68" becomes 48982.68. Anything else is unparsed.
  - For counts, take the first number in the string, so "1.00xITEMs" becomes 1, and "(Qty=4.00)" becomes 4. A separator followed by 1 or 2 digits is a decimal mark.
  - `discount_price` is sign-insensitive, and its gold lists both signs. No other field is sign-insensitive. For example, one receipt prints change as "-4,500" (cord-dev-013), and its gold stays -4500.
  - When `gt_parse` has several values for one key, they become a list of acceptable values, and any one of them counts. 5 keys have several values:
    - cord-test-013 subtotal_price and cord-test-058 cashprice repeat the same value, so each collapses to one gold.
    - cord-test-095 cashprice ["62,000", "201"], cord-test-099 tax_price ["6,818", "0"] and cord-dev-034 creditcardprice ["283,338", "5962"] become alternatives, which is lenient.
    - Counting the 4 discount sign pairs, 7 scalar golds are lists of alternatives.
  - The build never guesses an unparseable gold. It drops the field from that item and records it in `meta.dropped_fields`. There are 3 cases: tax "-" on cord-test-060 and cord-test-065, and discount "Discount(0%)" on cord-dev-029.
  - `meta.gold_raw` keeps the raw gold strings, and `meta.gt_parse` holds the full `gt_parse`. Two golds are not integers: cord-dev-077 tax 5272.73 and cord-test-041 service 48982.68.
- The list fields (`menu_nm`, `menu_cnt`, `menu_price`) are only for list-capable arms: Clairic, Haiku and the VLM. Sage 2 has no list type in the 20 Oct spec.
  - For `type: list`, `gold` is the list value itself (an ordered multiset of verbatim strings), not a list of alternatives. Null means no menu item carries that key.
  - `meta.list_gold_typed` holds the numeric parses of `menu_cnt` and `menu_price`.
  - The items do not ask for sub-items (`menu.sub`). The build joined one multi-line item name with a space (cord-dev-037).
  - Report list fields separately from the scalar Sage head-to-head (the coverage rule, suite design decision L9).
- `meta.gold_reachable_oracle` flags whether the verbatim gold string occurs in the oracle text (whitespace-collapsed, lower-cased). It does for 785/785 present scalar golds. `meta.gold_reachable`, measured on the harness OCR, stays null until that OCR exists (suite design decision C9).
- `meta.unit_hint` gives "IDR" for the money fields. Pass it as Clairic's `unit`. *(A critic found on 2026-10-23 that the harness had ignored it and sent `unit: "currency"`. `harness/common.apply_hints` now copies it to the money fields' `unit`.)*
- A critic fix on 2026-10-23, which changed only the harness, left `items.jsonl` unchanged. The money fields carried no `number_locale` and the list fields carried no `items`. As a result, the harness scored a verbatim "45.500" as 45.5, and it scored `menu_price` and `menu_cnt` as text even though their instructions ask for a number.
  - `harness/common.apply_hints` now sets `number_locale: dot_thousands` on the money and number-list fields. It also sets `items` from `meta.list_item_types`, mapping money to number. Both changes apply to requests and to scoring.
  - The effect is recorded in `../../scripts/critic_selfcheck_2026-10-23.out.json`. `meta.list_gold_typed` used as a prediction scores 1.0 on 379/379 list fields (152/379 before). The Indonesian "45.500" form of a money gold of at least 1,000 scores 1.0 on 675/675 (6/675 before).
- Every arm gets identical field instructions. Each one ends "If the receipt does not show it, return null."

### COMPARABLE view (test only, `meta.comparable`)
- The prompt is the CC-OCR prompt verbatim, with its per-image key set. There are 73 distinct prompts over the 100 images.
  - The schema lists only the keys present in that image's gold, so it leaks which keys are present (suite design decision L2).
  - All fields are text. The gold is the CC-OCR answer JSON, menu included.
- We checked the following:
  - CC-OCR `test_<i>.jpg` maps to CORD v2 test `image_id` i.
  - The CC-OCR answer equals the CORD v2 `gt_parse` on 100/100 images.
  - Image sizes match on 100/100.
  - CC-OCR's JPEG bytes differ from the CORD v2 PNG bytes on 100/100 because CC-OCR re-encoded them (largest 1.45 MB). The comparable arm therefore sends CC-OCR's own bytes, saved in `comparable/images/`.
- `comparable/label.json` and `comparable/qa.jsonl` follow CC-OCR's own layout (`raw/cc-ocr-github/assets/doc/data_structure_en.md`). We have not run the evaluator here, because it is downloaded code. Running it is a harness step.
- Only this score may sit next to the CC-OCR numbers below. Never print the TYPED and COMPARABLE scores in one table.

## Metric and why
- The TYPED view uses exact_norm on typed numbers (money and count). Clairic `number` and Sage `money`/`integer` return typed values, so a format difference such as "60.000" versus 60000 should not count as an error. List fields use list_f1.
- The COMPARABLE view uses CC-OCR field-level micro F1, because the published rows were scored that way.
  - The scorer flattens fields Donut-style, and F1 = TP / (TP + (FP + FN)/2).
  - Normalisation only converts full-width characters to half-width and normalises spacing. It does not lower-case.

## Published baselines (reference, not paired)
| system | score | setting | provenance | split / n | URL (opened) |
|---|---:|---|---|---|---|
| GPT-4o (2024-08-06) | 81.26 | CC-OCR Table 5 KIE field F1, image + CC-OCR JSON-schema prompt, zero-shot | Alibaba/Qwen authors (competitor-run) | CORD v2 test / 100 (gold verified identical) | https://arxiv.org/html/2412.02210v3 |
| Gemini-1.5-Pro (002) | 78.73 | same | same | same | same |
| Claude-3.5-Sonnet (20241022) | 71.51 | same | same | same | same |
| Qwen2-VL-72B | 69.02 | same | same (own model) | same | same |
| InternVL2-76B | 66.12 | same | same | same | same |
| GPT-4V +Image | 64.05 | LMDX Table 3, zero-shot (\|D\|=0), image + IE instruction, Micro-F1 (Donut eval tool) | Google (method authors) | CORD v1 official test | https://arxiv.org/abs/2309.10952 (v2; HTML https://arxiv.org/html/2309.10952) |
| Gemini Pro +OCR | 59.57 | same, raw OCR text, no layout | same | same | same |
| PaLM 2-S +OCR | 55.85 | same | same | same | same |
| GPT-3.5 +OCR | 48.92 | same | same | same | same |
| Gemini Pro +Image | 47.12 | same, image | same | same | same |
| LMDX Gemini Pro | 66.03 | same, OCR text + layout tokens. Its base extractor was tuned on a mixture that the paper says has "no document or schema" overlap with the targets | same | same | same |
| GPT-4+OCR | 58.3 | DocLLM Table 5, zero-shot, OCR text (no layout), extraction-template prompt, F1 | JPMorgan (method authors) | CORD test split; CORD version UNKNOWN | https://arxiv.org/html/2401.00908 (v1) |

- The CC-OCR API versions and dates come from its HF card: "GPT-4o-2024-08-06, Gemini-1.5-Pro-002, Claude-3.5-Sonnet-20241022", tested "around November 20th, 2024" (https://huggingface.co/datasets/wulipc/CC-OCR).
- LMDX used CORD v1. The paper links https://huggingface.co/datasets/naver-clova-ix/cord-v1 and says "we use the official train, dev and test splits". CORD v2 corrected labels, so the LMDX rows differ from ours in gold version as well as in protocol.
- Every row includes menu line items, which Sage 2 cannot type.
- Only the CC-OCR rows are comparable. The LMDX and DocLLM rows use their own prompts and OCR, so they are context only.
- Whether published CORD rows exist for Clairic, Sage 2 or Haiku 5.5 is UNKNOWN. We did not search beyond the papers above.
- Local copies of the paper pages are in `raw/baselines/`.

## Contamination risk: high
- CORD has been public since Dec 2019 (v0) and on HF since Jul 2022, with gold `gt_parse` for every split.
- It is the standard Donut fine-tuning demo, so the receipts and answers are very likely in web-scale training data.
- The CC-OCR TSV (Dec 2024) contains the exact test prompts and answers.
- CORD is not among the 127 tasksource extraction train sources (suite design decision L1).
- Read any vendor "win" here together with the results on the fresher sets (suite design decision L18).

## Tokens (o200k_base, tiktoken 0.9.0; `scripts/selfcheck.out.json`)
| quantity | p50 | p95 | max |
|---|---:|---:|---:|
| Oracle text (`meta.oracle_text`) | 55 | 128 | 196 |
| 12 field definitions (name + type + instructions, JSON) | 670 | 670 | 670 |
| Oracle text + field defs (a Sage-style text call) | 725 | 798 | 866 |
| CC-OCR prompt (test, comparable arm) | 185 | 202 | 217 |

- The harness OCR text will be longer than the oracle text, because it adds the store header, dates and payment lines. Treat the oracle counts as a lower bound.
- Image tokens for Clairic vision are UNKNOWN. Image files are 1.39 MB at p50, 2.27 MB at p95 and 3.72 MB at most, so all are within 4 MiB.
- According to the suite notes, Clairic re-bills the state for each field, so budget per field, not per receipt. Image-token billing is UNKNOWN.

## How text was produced
- No text ships in the items (`text` is null). The harness OCR adds the primary text later.
- The oracle text is the CORD human word transcript, described above and stored in `meta.oracle_text`.

## Risks and limits
- Confidence intervals are wide. Even n = 200, or 195 effective receipts, gives wide CIs. The planning half-width was about ±8 to 10 points per arm at n=100. A field-level, cluster-resampled bootstrap gives tighter intervals.
- The scalars are null-heavy. 56% of scalar slots have null gold, so abstention behaviour dominates the pooled number. Always report present-gold and null-gold results separately.
- Most gold leaves are menu line items (1,368 list entries against 785 present scalars). Sage 2 (20 Oct spec) has no list type, so the Sage head-to-head covers only the 9 scalars.
- A null gold marks a key that is absent from the annotation. The audit found 1 wrong null among 47 slots in the targeted total_price-null stratum and 0 among 60 random slots. The targeted verifier pass found 2 more wrong nulls and 1 wrong lenient value (4 adjudicated slots in all), plus 3 ambiguous slots.
- The photos are mostly cropped receipts with privacy blur (`dontcare` regions on 21/200 items), and models see the blur.
- The oracle text is not OCR. It has no distractors, so never report it as an OCR ceiling.
- The build re-encoded 28 images to JPEG to fit the 4 MiB cap. The comparable arm uses CC-OCR's JPEGs, not ours.

## Files
- `build.py` rebuilds `items.jsonl`, `img/`, `comparable/` and `build_report.json` from `raw/`. It also merges the two audit files (`audit/null_gold_audit-2026-10-23.json`, `audit/verifier_gold_audit-2026-10-23.json`) if they are present.
- `scripts/selfcheck.py` writes `scripts/selfcheck.out.json` (format checks with 0 errors, and token counts).
- `scripts/null_gold_audit.py` writes `audit/null_gold_audit-2026-10-23.json`.
- `scripts/peek_gold.py` is the gold-structure peek we used while designing the set.
- `raw/` holds:
  - the parquet files;
  - HF and GitHub metadata and licence copies;
  - `ccocr/CORD_100.tsv` and `ccocr/CC-OCR_README.md`;
  - `cc-ocr-github/` (the evaluator at the pinned commit);
  - `baselines/` (paper pages as fetched).
