# docvqa: DocVQA single-page, validation (proportional and handwritten strata)

Built 2026-10-23 for extract-bench at a cost of $0. The build called no metered API.
The use case is reading one value off a scanned business page (forms, tables, letters, figures). The sample oversamples handwriting.
The modality is `image`. Each item has one page PNG plus the dataset's official OCR text. Clairic-vision gets the image, and Clairic-text and Sage 2 get the identical OCR text.

## 1. Source and version
| What | Where | Revision / version |
|---|---|---|
| Paper | Mathew, Karatzas, Jawahar, *DocVQA: A Dataset for VQA on Document Images*, WACV 2021. https://arxiv.org/abs/2007.00398 (PDF: `refs/docvqa_2007.00398.pdf`) | arXiv v3 |
| Dataset page | https://www.docvqa.org/datasets (`refs/docvqa_org_datasets.html`). The images are pages from the UCSF Industry Documents Library, the OCR is "a commercial OCR solution", and the files are on the RRC portal behind a login | n/a |
| Mirror A: questions, answers and official OCR | https://huggingface.co/datasets/pixparse/docvqa-single-page-questions | commit `33136ef456fa5a3fe68568d6e31dda4eeff95b9b` (lastModified 2024-03-29). Files `data/validation-0000{0..3}-of-00004.parquet` (1.55 GB in total) |
| Mirror B: question_types and page PNGs | https://huggingface.co/datasets/lmms-lab-encoder/DocVQA (config `DocVQA`) | commit `539088ef8a8ada01ac8e2e6d4e372586748a265e` (lastModified 2024-04-18). Files `DocVQA/validation-0000{0..5}-of-00006.parquet` (1.06 GB in total) |

- We downloaded the following into `raw/`, all by range reads, without saving any full parquet:
  - The non-image columns of both mirrors' val split: `pixparse_val_meta.parquet` (6.7 MB, including the full OCR with words, boxes and confidence) and `lmms_val_meta.parquet` (0.25 MB).
  - The original PNG bytes of the 283 sampled pages: `lmms_png/<docId>.png`, plus `manifest.json` with each file's sha256, source file, row group and questionId.
  - Both dataset cards (`README_*.md`) and the HF API metadata (`hf_api_*.json`).
- The build range-read the 52 lmms row groups that hold a sampled page into a temporary sparse file, and it deleted that file once it had extracted the pages.

## 2. Licence
- The pixparse mirror card says `license: mit`. Its README "Licensing Information" section also says "MIT" (`raw/README_pixparse.md`, at the pinned commit).
- The lmms-lab-encoder mirror card says `license: apache-2.0` (`raw/README_lmms.md`).
- The original RRC terms are UNKNOWN.
  - docvqa.org says to "read the terms and conditions listed in the Download page" on the RRC portal, which needs a login.
  - `https://rrc.cvc.uab.es/?ch=17&com=downloads` timed out on 2026-10-23 (curl exit 28 after 30 s).
- The page images are scans from the UCSF Industry Documents Library.
- This repo does not redistribute the page images. See `LICENSE-DATA.md`.

## 3. Population and sample (seed 20)
- The population is DocVQA val, with 5,349 questions on 1,286 pages. Both mirrors have exactly these questionIds. Question text, docId, UCSF id/page, image filename and split agree on all 5,349. On 206 questions the answer *lists* differ, but only in order, so the answer *sets* are identical.
- Question types come from mirror B. They are multi-label: 4,282 questions have 1 tag, 993 have 2, 71 have 3 and 3 have 4.

  | Tag | val | Expected in P | P (240) | All 350 |
  |---|---:|---:|---:|---:|
  | layout | 1,981 | 88.9 | 93 | 118 |
  | table/list | 1,780 | 79.9 | 81 | 98 |
  | form | 1,021 | 45.8 | 42 | 110 |
  | free_text | 765 | 34.3 | 36 | 40 |
  | handwritten | 319 | 14.3 | 14 | 124 |
  | figure/diagram | 265 | 11.9 | 13 | 13 |
  | others | 236 | 10.6 | 11 | 14 |
  | Image/Photo | 98 | 4.4 | 4 | 4 |
  | Yes/No | 28 | 1.3 | 1 | 5 |

- The stratification key is `primary_type`. It is `handwritten` if the question carries that tag, and otherwise the question's rarest tag by val-wide frequency. This keeps rare tags (Yes/No, Image/Photo, others) at their val share, which plain proportional allocation cannot do for a multi-label set.
- Stratum P (proportional, 240) uses largest-remainder allocation of 240 over `primary_type`, with simple random sampling within each stratum.
  - The allocation is table/list 73, layout 60, form 35, free_text 31, handwritten 14, figure/diagram 12, others 10, Image/Photo 4, and Yes/No 1.
- Stratum H (handwritten extra, 110) is a simple random sample of 110 from the 305 handwritten questions not already in P.
  - The handwriting slice is then 124 of the 319 handwritten val questions, on 81 pages.
  - val has 319 handwritten questions on 126 pages, which matches `../../survey/final-review-2026-10-23/docvqa_val_handwritten.out.json`.
- `items.jsonl` lists the P block first, then the H block, each shuffled with seed 20. Any prefix of the P block is therefore a proportional val sample.
- `meta.weight_val` reweights the sample by post-stratification on the handwritten flag.
  - The weights are 2.5726 for handwritten items (319/124) and 22.2566 for the rest (5,030/226). They sum to 5,349.
  - Use the weights, or the P block alone, whenever you quote a number next to a published DocVQA score. Report H separately.
- The 350 questions sit on 283 pages, so several questions share a page. Use `meta.doc_id` for a cluster (page-level) bootstrap.

## 4. Field mapping
- Each item has one field: `{"name": "answer", "type": "text", "instructions": <the DocVQA question verbatim>, "gold": [<all acceptable answers>]}`.
  - The gold is the annotators' answer list, de-duplicated with order kept.
  - There are no null golds. DocVQA has no unanswerable questions, so this set does not test abstention.
- Sage 2 runs in bare-question mode (`extract` = the question string). This is a declared adaptation (suite design decision C2).
- Clairic gets one `text` field whose `instructions` are the question.
- For both, text answers are spans of the input text. This matters because ANLS gives partial credit (see §6). A null prediction scores 0 under ANLS, so report the null rate as its own number.

## 5. How the text was produced
- `text` holds the official DocVQA OCR from mirror A's `ocr_results`: the line texts, in the order shipped, joined with `\n`. Each item sets `text_source = official_ocr`.
- Word boxes, per-word "Low" confidence flags, page size and orientation stay in `raw/pixparse_val_meta.parquet`. Per-item counts are in `meta.ocr`.
- In our judgement, the JSON layout (`clockwise_orientation`, `unit: pixel`, 8-point boxes, "Low" confidence) looks like Microsoft's Read OCR output. The source only says "a commercial OCR solution".
- `meta.gold_reachable` is true when some gold answer, lower-cased and whitespace-collapsed, is a substring of the OCR text normalised the same way. This is the verbatim-span ceiling for the text arms.
  - On full val it is 85.64% of 5,349. This exactly equals the paper's "OCR substring UB" for val (85.64%, Table 1).
  - The survey's 82.8% did not collapse whitespace, so it missed answers that wrap across OCR lines.
  - In the sample it is 75.7% of all 350 items, 87.9% on the P block, 51.6% on handwritten items (n=124) and 88.9% on non-handwritten items (n=226).
  - `meta.gold_reachable_alnum` is looser, because it compares alphanumerics only (full val 92.28%).
- The pre-registered reporting (suite design decisions L15 and C9) compares text against vision on the reachable subset and on the full set. It also compares handwritten against non-handwritten items.
- The official OCR is much weaker on handwriting. Gold is reachable on only about half the handwritten items. The OCR flags 23.1% of words on the handwritten items' pages as "Low" confidence, against 12.3% across all items. On that stratum, the text arms (Sage 2, Clairic-text) are therefore capped well below Clairic-vision. This is a property of the input, not of the readers.
- The build runs mirror alignment checks and aborts on any failure (see §9 for the deviation from the spec):
  1. Question text, answer set, docId, UCSF id/page and image filename are identical across the two mirrors (5,349/5,349).
  2. The build re-checks the sampled PNG's sha256 against `manifest.json`, and the PNG filename equals the mirror-A filename.
  3. PNG width/height equals the OCR page width/height (283/283 pages).
  4. A content check passes a page if any one of three tests passes:
     - (a) Shift profile. With the OCR word boxes shifted vertically from −60 to +60 px, the darkness inside the boxes peaks within ±8 px of 0. This passes 270/283 pages.
     - (b) Local tesseract 5.5.0 OCR (free and local, cached in `checks/tesseract/`). It recovers at least 25% of the official OCR's distinctive words, meaning words of at least 4 characters, ignoring the 193 words found on at least 5% of pages. This passes 254/283. Either (a) or (b) passes 277/283.
     - (c) By eye. We checked the last 6 pages visually against their OCR (`checks/manual_verified.json`). They are 8883, 9031, 9048, 9072, 13885 and 14179. They show handwriting, rotated text and a grey graphic page.
   - As a control, we tested each page's OCR against the next (wrong) page image. That pairing passes (a) 11.7% of the time, (b) 1.4% of the time, and either 13.1% of the time.
   - Checks 1 to 3 therefore carry most of the weight. 169/283 sampled pages have a width × height that no other val page has.
   - Per-page results are in `meta.image.alignment`.
- The images are the original PNG bytes from mirror B, copied unchanged to `img/<docId>.png`. The build re-encoded 0 of 283. The largest file is 2.56 MiB. Every image is 1 page and at most 4 MiB, and there is 1 image per item.

## 6. Metric: ANLS
- ANLS (Average Normalized Levenshtein Similarity) uses τ = 0.5.
  - Each question scores the maximum over gold answers of (1 − NL) if NL < 0.5, and 0 otherwise. NL is the normalised Levenshtein distance between prediction and gold.
  - The ST-VQA paper does not specify case handling. Follow the common implementation, which lower-cases, strips and collapses whitespace on both sides, as lmms-eval's `anls` does (https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/api/metrics.py, `main` branch read 2026-10-23, commit UNKNOWN; copy in `refs/lmms_eval_metrics.py.txt`).
  - lmms-eval normalises the strings for the edit distance but divides by the length of the *un-normalised* strings (`max(len(answer), len(pred))`). The extract-bench scorer must pick one and state it. In our judgement, it should use the normalised lengths.
  - In one edge case the two rules differ. lmms-eval zeroes a score only below similarity 0.5, so NL = 0.5 exactly scores 0.5 there and 0 in the paper. Use the paper's rule (NL < τ) and say so.
  - The score is the mean over questions.
  - The definition and τ = 0.5 come from Biten et al., *Scene Text VQA*, https://arxiv.org/abs/1905.13648 (`refs/stvqa_anls_1905.13648.txt`, about lines 353 to 387).
  - DocVQA adopts ANLS as its primary metric (paper §5.1).
- We use ANLS because it is the official DocVQA metric and every published baseline below uses it. Report exact-match accuracy as a secondary number.
- The weighted aggregate is Σ w·ANLS / Σ w, using `meta.weight_val`.

## 7. Published baselines (reference only, not paired)
Our items are a 350-question val sample, and most published numbers are on the hidden-gold test split. The published numbers are for orientation only. The paired comparisons are our own arms on identical items.

| System | Score | Setting | URL (opened 2026-10-23) |
|---|---:|---|---|
| Claude 3.5 Sonnet | 95.2 | DocVQA test (row label "DocVQA_test"), image input, as reported in the Qwen2.5-VL report, Table 5. The report never names the metric (the word ANLS does not appear). We infer ANLS×100 from the DocVQA test server's official metric | https://arxiv.org/html/2502.13923 |
| Gemini 1.5 Pro | 93.1 | same table | https://arxiv.org/html/2502.13923 |
| GPT-4o | 91.1 | same table | https://arxiv.org/html/2502.13923 |
| InternVL2.5 78B | 95.1 | same table | https://arxiv.org/html/2502.13923 |
| Qwen2.5-VL 72B / 7B / 3B | 96.4 / 95.7 / 93.9 | same table | https://arxiv.org/html/2502.13923 |
| Human | 0.981 ANLS (94.36% acc) | test; DocVQA paper Table 1 | https://arxiv.org/abs/2007.00398 |
| bert-large-squad, fine-tuned on DocVQA, with the official OCR text as input | val 0.655 ANLS (54.48% acc); test 0.665 | DocVQA paper Table 3. This is the closest published analogue of a text-only reader on this OCR | https://arxiv.org/abs/2007.00398 |
| bert-base, fine-tuned on DocVQA, OCR text | val 0.556; test 0.574 | Table 3 | https://arxiv.org/abs/2007.00398 |
| M4C best (dynamic vocab 500) | val 0.385; test 0.391 | Table 2 | https://arxiv.org/abs/2007.00398 |
| OCR substring upper bound | val 85.64% acc; test 87.00% | Table 1 (answer is a substring of the serialised OCR) | https://arxiv.org/abs/2007.00398 |

- The Qwen2.5-VL table does not say whether Alibaba ran each competitor row or copied it from vendor reports, so this is UNKNOWN.
- A verifier note from 2026-10-23: all Table 5 numbers above were re-checked against https://arxiv.org/html/2502.13923 and all DocVQA-paper numbers against Tables 1 to 3 of https://arxiv.org/pdf/2007.00398; they match. The only change was to the metric wording on the Qwen rows, which is inferred, not stated.
- Per-question-type (handwritten) numbers are not published as text. The 2020 challenge report (https://arxiv.org/abs/2008.08899) shows them only as a figure (Fig. 2).
- No Clairic or Sage DocVQA number has been published. We searched the local copies of the Clairic docs and blog and the Sage 2.0 launch draft, and found no mention.

## 8. Contamination risk
In our judgement, the contamination risk is high.
- Val gold has been public since 2020.
- `lmms-lab-encoder/DocVQA` is the top result by downloads in the HF dataset search for "docvqa", with 45,515 downloads per the HF API listing on 2026-10-23.
- DocVQA train is in common VLM tuning mixes. Whether val is too is UNKNOWN per model.
- A high score here says little about fresh documents. Read it next to RealDoc (2026) and the other fresher sets in the suite.

## 9. Deviations from the spec, declared
1. Metadata and content checks replace the image-hash join (§5 checks 1 to 4).
   - The spec asked to join the two mirrors "on questionId + image hash". Hashing mirror A's images needs mirror A's image bytes.
   - Each pixparse val file is a single row group, so its image column chunk is 323 to 429 MB and cannot be range-read per page. Fetching it means pulling 1.55 GB.
   - Measured HF throughput today was 0.05 to 0.35 MB/s (shared link), so we did not do that fetch.
   - The build records the sha256 of every page used (`meta.image.sha256_original`). The byte-level cross-mirror check can be added later by streaming the pixparse image column.
2. Multi-label proportional sampling uses the `primary_type` key (§3). The realised tag shares are in the table in §3.

## 10. Size and tokens
- There are 350 items, 350 fields and 0 null golds.
- The official-OCR text has these o200k token counts per item: p50 298, p95 814, max 1683, and mean 339.3. With the question added, the counts are p50 304, p95 828.
  - Across all of val, the counts are p50 326, p99 1,228, max 1,905 (`../../survey/ocr-raw/fit_docvqa_val.json`).
  - Every item is far below Sage 2's 16,384-token limit and Clairic's 65,536 context.
- The images are 283 PNGs, 168.4 MB in total, with a p50 of 190 KB per page.
- `raw/` is about 168 MB.

## 11. Files
- `items.jsonl`: the 350 items.
- `img/`: 283 page PNGs.
- `build.py`: stages `fetch-meta`, `fetch-images` (network) and `build` (offline; re-creates `items.jsonl` and `img/` from `raw/`).
- `build_report.json`: design, allocation, tag marginals, reachability and alignment stats.
- `selfcheck.out.json`: format validation and token stats.
- `scripts/`: exploration and self-check scripts (`selfcheck.py`).
- `refs/`: copies of the papers and pages cited.
- `checks/`: the tesseract cache and the manual-verification log.
- `fetch_images.log`.
- To reproduce, run the command below. It needs `tesseract` on PATH, which only the alignment check uses, and it caches the results in `checks/tesseract/`.
  `PYTHONDONTWRITEBYTECODE=1 /opt/homebrew/bin/uv run --no-project --with pyarrow --with fsspec --with aiohttp --with requests --with pillow --with tiktoken --with numpy python -I build.py build`
  - The output is byte-identical across reruns. We checked the sha256 of `items.jsonl` twice.
  - `fetch-meta` and `fetch-images` re-download from the pinned revisions. That is about 25 MB plus about 1.0 GB of range reads, or roughly 50 minutes at the 2026-10-23 link speed.
