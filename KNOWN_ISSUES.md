# Known issues (v0.1)

Read these before you quote a number. Each is a limitation of v0.1, and the results don't hide any of them.

## Design

1. One system defines the item selection. The v0.1 pool keeps the documents that fit in one Sage 2 request
   (`splits/v0.1.json`), which removes the long Kleister-Charity reports. Other systems may do relatively better on
   those reports. We plan a rule that doesn't depend on any system, or a separate long-document board.
2. Only Sage 2 was tuned. Levanto fitted its abstention thresholds on the `fit` half, and every other system ran with
   default settings. The headline uses the `eval` half, so the board never scores Sage 2 on items it was tuned on, but
   the tuning budget still isn't equal.
3. The sealed holdout isn't sealed. The v0.1 runs left out each dataset's 20% holdout (`common.is_holdout`), but those
   items are in `items.jsonl` in this repo, so treat them as public.
4. All 12 datasets have been public for years. Whether any system, Sage 2 included, trained on them is UNKNOWN. There
   is no private test set and no canary string yet.
5. Document-AI vendors built two of the sets: OmniAI built OmniAI OCR and Extend built RealDoc. Their cards say so.

## Statistics

6. Each system ran once, so v0.1 doesn't measure how much an API model's answers vary between runs.
7. The intervals are item-level. They don't cluster by template (FATURA has 10) or by charity (Kleister-Charity), and
   they aren't corrected for comparing across 10 datasets. The per-dataset ✱ marks are therefore optimistic. The
   package has `score.paired_bootstrap` (clustered) and `score.holm` for a stricter analysis.
8. Some datasets are small. The eval half has 24 items for TAT-QA, 31 for Kleister-NDA, 44 for Kleister-Charity and
   45 for Emails, so a difference of 10 points or more there can be noise. Check the intervals in
   `results/v0.1/board.md`.

## Data

9. The redistribution terms are UNKNOWN for Kleister-NDA, Kleister-Charity, VRDU, DocVQA (at the source), SROIE (at
   the source) and the RealDoc PDFs. See [LICENSE-DATA.md](LICENSE-DATA.md).
10. Submitting the gold values as predictions scores 100% on FATURA, Emails, Kleister-NDA, SROIE and SQuAD 2.0, and the
    tests check this. On VRDU and RealDoc a few gold values fail their own normaliser: VRDU list fields whose gold is a
    list of alternative readings, and one RealDoc gold value of `"( )"`. The effect is under 0.5 points per set.
11. OCR caps the receipt sets. On SROIE and CORD the shared Tesseract text misses much of the gold, so the board keeps
    them out of the mean.
12. The repo doesn't ship page images. You can re-score the image board (`clairic_img`) from its predictions, but
    re-running it needs the images, which each `build.py` can fetch.

## Systems

13. The Sage 2 row is a preview: "Sage 2 rc9 (preview), fast tier, 28 Oct 2026". Its extraction price isn't public,
    so the board shows no cost.
14. Each open-weight model ran with one prompt format, the one its model card documents, and with no prompt search.
    Other formats may score higher. Schematron was built for HTML pages, so we wrap each document's text in minimal
    HTML for it. GLiNER2.5 tags spans, so it can't return a value that isn't verbatim in the text. That caps it on the
    question sets and on computed values.
