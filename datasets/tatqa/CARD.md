# tatqa: TAT-QA test, proportional sample [Tier C]

This set tests questions over a financial-report table and its paragraphs, taken from 10-K and annual-report pages. Answers are spans, multi-span lists, or computed numbers (arithmetic or counts) with a scale (thousand, million or percent). It is the suite's only test of Clairic's marketed "computed values".

This set is Tier C, so it is reported apart and never in the headline. The tasksource Clairic-style `extraction` corpus contains TAT-QA text-span rows from train and dev contexts, and the test gold has been public since 2024. It is also not an OCR test (suite design decision C15).

| | |
|---|---|
| Items | 150 questions (one field `answer` each) from 120 distinct table-and-text contexts |
| Groups (reported separately, never averaged) | Extractive: span 64 (all arms) and multi-span 19 (list-capable arms only). Computed: arithmetic 63, count 4 (Clairic and Haiku only; Sage 2 is n/a) |
| Modality | Text (`text_source: native`). There are no images, so page count is n/a (`meta.pages: null`) |
| Metric | `numeric_tol` (ours), plus the official scale-aware `tatqa_eval` EM/F1 via `comparable/run_official.py`. The official score is the comparable row and has two scale bounds |
| Null gold | 0 of 150. TAT-QA has no unanswerable questions, so this set does not test abstention |
| Sealed holdout | 31 items (20% per answer type, seed 20, `meta.holdout`) |
| o200k tokens per item (text) | p50 397; p95 935; max 1,532. Text plus instructions: p95 961. That is far below both Sage 2's 16,384 cap and a single Clairic 8,192-token pass |

## Source and version
- The repo is https://github.com/NExTplusplus/TAT-QA, pinned at commit `870accc41953dcde885aabeb963d94aabdc0fbc3` (HEAD, 2024-12-09, "Update index.html"; from the GitHub API `repos/NExTplusplus/TAT-QA/commits`).
- The test gold file, `dataset_raw/tatqa_dataset_test_gold.json`, arrived in commit `12fd7c8bb474e8b45e9f54dbf7c281c178f1a510` (2024-02-14, "add gold tatqa test set"). Its git blob is `3d5d3b691af572646ba9b1ac4d8f78c5dca09732`, which matches the GitHub contents API, and its sha256 is `c4d08418359c1d76468dec420ee748a37f48c06b63cb8ec2766f19d5d314b597`. The README says "Jan 2024: We release the ground truth for the TAT-QA Test set".
- The paper is Zhu et al., "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance", ACL 2021, https://arxiv.org/abs/2105.07624.
- We downloaded these files into `raw/`. All are free from GitHub or Hugging Face, and together they take 21.5 MB, including 5.4 MB of tasksource rows. The arXiv PDF is in `refs/`.
  - `tatqa_dataset_test_gold.json` (2.17 MB) is the evaluation source.
  - `tatqa_dataset_train.json` and `_dev.json` serve only to flag paragraph overlap. Their sha256s are `2df6e722…` and `8da095a8…`. All three files are byte-identical to the survey copies in `survey/final-review-2026-10-23/raw/`.
  - The repo's `README.md`, `LICENSE` and `leaderboard_index.html` (which is `docs/index.html` at 870accc4) are also there.
  - `official_eval/*.txt` holds `tatqa_eval.py`, `tatqa_metric.py`, `tatqa_utils.py`, `tatqa_metric_test.py`, `sample_prediction.json` and `requirement.txt` at 870accc4, kept as text.
  - `tasksource/tatqa_text_span_rows.json` and `tasksource/parts/` hold every `tatqa/text-span` row of `tasksource/tasksource-jev-typed-decisions` (config `extraction`, HF sha `8b1afcc6ad12554b347134a91b5f0dc6ae2361db`, lastModified 2026-10-21T23:44:21Z; fetched 2026-10-22T23:04Z UTC). `scripts/fetch_tasksource_tatqa.py` fetched them with column range reads of `refs/convert/parquet`. The script docstring lists the abandoned attempts.
- The full test split has 277 contexts and 1,663 questions. We counted these from the gold file, and `build.py` asserts the answer-type counts.
  - The answer types are span (714), arithmetic (699), multi-span (210) and count (40).
  - The answer sources are table (736), table-text (546) and text (381).
  - The scales are none (836), thousand (319), percent (294) and million (214).
  - There is a discrepancy, and its cause is UNKNOWN. TAT-LLM's Table 1 (arXiv 2401.13223) lists 1,669 TAT-QA test questions, but the released gold file has 1,663.
    - A verifier note from 2026-10-23 adds that the original TAT-QA paper's Table 1 (https://arxiv.org/abs/2105.07624, PDF opened) also gives 278 tables and 1,669 questions for test, while the released gold has 277 contexts. So the release is short by exactly one context and 6 questions. In our judgement, the public gold probably leaves out one test context. Which one is UNKNOWN.

## Licence
- The data is under CC BY 4.0. README line 105 says "The TAT-QA dataset is under the license of Creative Commons (CC BY) Attribution 4.0 International" (https://creativecommons.org/licenses/by/4.0/). The leaderboard page (https://nextplusplus.github.io/TAT-QA/) says the same. Attribution means citing Zhu et al. 2021.
- The code and scorer are under MIT (`raw/LICENSE`, "Copyright (c) 2021 Fengbin Zhu"), and the GitHub API also reports MIT.
- The tasksource repo's HF licence tag is `license:other`. We keep only its TAT-QA-derived rows, for the contamination check.

## Contamination exclusion (run before sampling; from the spec and suite design decision L1)
- The tasksource corpus holds only `tatqa/text-span` rows. That is 1,507 train rows, 188 validation rows and 0 test rows (source frequencies are in `survey/critique-legitimacy-2026-10-23/raw/tasksource_extraction_stats_*.json`).
  - A row's `state` holds only a TAT-QA context's paragraphs, with no table, joined by blank lines.
  - We checked this. 1,675 of 1,695 states equal a TAT-QA context's whitespace-normalised paragraph text exactly. In tasksource train, 1,487 match TAT-QA train contexts and 1 matches dev. In tasksource validation, 187 match dev. 0 match test, and 20 rows match no context exactly (see `scripts/tasksource_overlap_stats.out.json`).
  - Their `metadata.question_ids` match 0 of their 2,981 TAT-QA question uids.
- The rule drops a test context, with all its questions, if any of these holds:
  - (a) Its whole paragraph text is a substring of some tasksource state.
  - (b) Every paragraph of at least 20 characters appears inside some state.
  - (c) A question uid matches.
- Rule (b) dropped 1 context (6 questions). Its 3 paragraphs ("Consolidated income statement", "For the year ended 31 March 2019" and an IFRS 9/15/16 adoption note) all occur inside 2 tasksource train states, and the last two are also TAT-QA train paragraphs. No context met (a) or (c). After the drop, the eligible pool is span (711), multi-span (209), arithmetic (697) and count (40).
- Items with partial overlap (some but not all paragraphs) stay in, flagged in `meta.tasksource.partial_overlap`. That is 47 of 150 items (span 21, multi-span 5, arithmetic 17, count 4). Typically tasksource covers 1 of 2 to 15 paragraphs, and TAT-QA itself reuses 54 of the 63 covered paragraphs in train/dev. Report the clean slice of 103 items, with no paragraph in tasksource, as a robustness check next to the full 150 sampled items.
- TAT-QA also reuses text across its own splits. 42 sampled items share at least one paragraph of 20 or more characters with TAT-QA train/dev (`meta.tatqa_train_dev_para_overlap`). Any model fine-tuned on TAT-QA train has seen that text.
- We did not check non-TAT-QA tasksource sources, so their overlap is UNKNOWN. Sources such as `finer-139` and `finer-ord` (10-K sentences, 10,000 and 2,056 train rows) could share single sentences with these reports. In our judgement, they cannot contain a full table-and-paragraph context, so they don't meet the drop rule, but they are sentence-level exposure.

## Sampling (seed 20)
- The allocation is proportional to the full test set's answer types, by largest remainder. It assigns 64 items to span, 63 to arithmetic, 19 to multi-span and 4 to count, 150 in total.
- The draw runs `random.Random(20).sample(sorted(eligible uids), k)` for each stratum, in the order span, multi-span, arithmetic, count.
- For the holdout, a second `random.Random(20)` draws `round(0.2·k)` per stratum: 13 span, 4 multi-span, 13 arithmetic and 1 count, which is 31 in total.
- One item is one TAT-QA question. `meta.doc_id` holds the table uid and is the bootstrap cluster, since several questions can share a context.
- By answer source, the sample has table (67), table-text (53) and text (30) items. Arithmetic items split by scale into percent (28), thousand (17), million (11) and none (7). Span items split into none (45), thousand (12) and million (7). Multi-span items split into none (14) and thousand (5). The 4 count items all have scale none.

## Text (how it was produced)
- `text` is `"Table:\n"`, then the TAT-QA 2-D table as a markdown table, then `"\n\nText:\n"`, then the paragraphs in TAT-QA `order`, separated by blank lines.
  - Row 0 is the header. Newlines inside 4 test-set cells become spaces. No cell or paragraph contains `|`.
  - Cell and paragraph text is otherwise verbatim, so span answers stay spans of the state, with offsets.
  - TAT-LLM used the same composition (the table as markdown, plus the paragraphs) for its GPT-4 zero-shot row (arXiv 2401.13223 §3).
- Every span gold (64/64) and every multi-span gold (19/19 items) occurs verbatim and case-sensitive in `text`, so the extractive span ceiling is 100%. A dry-run draw had 2 golds that differed only by an initial capital, and the scorer casefolds.
- `text_for_sage2` is native, so Clairic-text, Sage 2 and Haiku read the identical string.

## Field mapping (`fields[0]`, name `answer`)
| TAT-QA answer type | Our type | Gold | Hints sent | Arms |
|---|---|---|---|---|
| span | `text` | `[official answer string]` | None. The instructions are the question | All |
| multi-span | `list` | Official answer list. `items: "number"` if every answer is a plain number (12 of 19), otherwise `"text"` | `items` | List-capable (Clairic, Haiku). Sage 2 has no list type, so it is unscored automatically |
| arithmetic | `number` | `[value in the report's units, fully expanded value]`. The expanded value multiplies by 1e3, 1e6 or 1e9 for thousand, million or billion, or is value/100 for percent | `decimals: 2`; no `unit` | Clairic, Haiku. Sage 2 is n/a (`na_arms: ["sage2"]`, `computed: true`) |
| count | `number` | `[int]` | `integer: true` | Clairic, Haiku. Sage 2 is n/a |

For arithmetic items, the instructions are the question plus "Compute the value from the table and text. Give a percentage as the percent number (12.5 for 12.5%) and an amount in the units the document uses for it (1234 when the table shows 1,234 in thousands). Round to 2 decimal places."
- `decimals: 2` is not a leak. 680 of 681 percent-free test derivations satisfy gold = `round(derivation, 2)`, and the official scorer itself rounds to 2 decimal places (`get_answer_str`). So 2 decimal places is the dataset convention, not per-item information.
- We send no `unit`, because the scale would hand over part of the answer. Instead, the scorer accepts both the report-unit number (TAT-QA's convention) and the expanded number.
- Sage is n/a on computed items. The 20 Oct `/v1/extract` spec returns verbatim spans with offsets and has no compute capability, so computed cells show n/a, not 0.
  - The harness does not read `na_arms` yet, so whoever builds the table must drop Sage rows where `meta.group == "computed"`. (This was a TODO for the harness owner; I did not edit `harness/`.) *(Resolved on 2026-10-23. `clients.na_for` marks these fields unscored for Sage, `run.py read` records an all-n/a item as n/a with no call, and `estimate.py` skips them.)*
  - If the new Sage version adds computed fields, re-enable the arm, and say so.

## Metric and why
- `numeric_tol` is our metric, and we report it per group.
  - A number matches if it is within max(0.005, 1e-4·|gold|) of any acceptable gold. Otherwise the scorer falls back to case-folded text equality, which covers span answers.
  - Lists use the harness `list_f1`, which compares normalised items as a multiset.
  - In the self-check, every acceptable gold scores 1.0, and a 1% perturbation scores 0 (`scripts/selfcheck.out.json`).
- The official `tatqa_eval` gives the comparable row. `comparable/run_official.py` feeds an arm's predictions to the unmodified official `TaTQAEmAndF1`.
  - The vendored copies in `comparable/official/` are byte-identical to `raw/official_eval/*.txt`. Their sha256s are `a4110626…` (eval), `2aeeac47…` (metric) and `a84bb2f9…` (utils).
  - The gold is `comparable/tatqa_gold_sample.json`, which holds the 150 sampled questions in the official structure. At run time the script restricts it to the uids the arm has records for.
  - The official EM requires a predicted scale, and no arm has a scale output channel. So the script prints two bounds:
    - The oracle bound sets pred_scale to the gold scale. It is the upper bound. The instruction asked for report units, which is what TAT-QA's `scale` annotates.
    - The none bound sets pred_scale to "" and is the lower bound.
  - Published systems predicted the scale themselves, so neither bound is strictly comparable. Quote both.
    - The verifier corrected this on 2026-10-23. It holds for the fine-tuned rows (TagOp, TAT-LLM), but not for the GPT-4 and GPT-3.5-Turbo zero-shot rows. TAT-LLM §4.1 (arXiv 2401.13223) says: "For zero-shot prediction using LLMs, we omit the scale prediction and compare the predicted value with the ground truth value only, assuming the scale prediction is always correct." That is the same protocol as our oracle bound, so oracle is the like-for-like bound against those two rows. The remaining differences are that those rows pool all answer types over 1,669 questions and use a generative prompt (best of 3 templates).
  - In the selftest (`comparable/selftest.out.json`), gold-as-prediction scores 100.0 EM under oracle and 46.67 under none. Expanded or fraction answers score 84.0 (arithmetic 100) under none. A wrong answer scores 0.
- The official script has a quirk. Its `if not prediction` check scores a raw numeric `0` as wrong. The official `sample_prediction.json` passes raw numbers, so published systems may have hit this. Our adapter passes strings, so it credits a correct `"0"`. No gold in the final sample is 0 (a dry-run draw had one, scoring 99.33 with raw numbers), so the quirk does not affect this set.

## Published baselines (TAT-QA test, all 1,663/1,669 questions pooled; reference only, not paired with our 150)
| System | Score (EM / F1) | Setting | URL |
|---|---|---|---|
| Human expert | 84.1 / 90.8 | Test; leaderboard and both papers | https://nextplusplus.github.io/TAT-QA/ |
| TAT-LLM (70B) | 81.4 / 88.4 | Test; LLaMA 2 fine-tuned on TAT-QA/FinQA/TAT-DQA train (leaderboard, 20 Jan 2024) | https://nextplusplus.github.io/TAT-QA/ |
| TAT-LLM (7B) | 74.56 / 82.88 (paper Table 3); 76.4 / 85.1 (leaderboard) | Test; fine-tuned. A verifier correction: these are two different models. In paper Table 3, 74.56 / 82.88 is TAT-LLM (7B) "fine-tuned with respective training set" (TAT-QA only), and TAT-LLM_All (7B) "fine-tuned with a combination of training sets" scores 76.49 / 85.13. The leaderboard's 76.4 / 85.1 matches TAT-LLM_All, as does the 70B row (81.42 / 88.49 is TAT-LLM_All 70B) | https://arxiv.org/abs/2401.13223 · https://nextplusplus.github.io/TAT-QA/ |
| GPT-4 | 71.92 / 79.71 | Test; zero-shot, best of 3 templates (end-to-end, chain-of-thought, step-wise; App. A/D), temperature 0, Sep-2023 API version, table as markdown plus text. Scale prediction omitted (assumed correct), which matches our oracle bound | https://arxiv.org/abs/2401.13223 (Table 3) |
| GPT-3.5-Turbo | 59.47 / 68.11 | Same as GPT-4 | https://arxiv.org/abs/2401.13223 (Table 3) |
| TagOp (original baseline) | 50.1 / 58.0 | Test; fine-tuned RoBERTa tagger plus operators | https://nextplusplus.github.io/TAT-QA/ · https://arxiv.org/abs/2105.07624 (abstract: 58.0 F1, human 90.8 F1) |

- We read the leaderboard rows from the pinned `docs/index.html` (`raw/leaderboard_index.html`) and the paper rows from `refs/arxiv_2401.13223v3.{pdf,txt}` (via pdftotext).
- In our judgement, comparability has two limits.
  - The published rows pool all answer types, while we report groups separately. To set ours beside them, reweight our per-type official EM to the full-test mix (714/699/210/40) and show the oracle and none bounds.
  - The published LLMs were generative with free-form reasoning. Clairic and Sage are single-pass typed extractors, so the GPT-4 zero-shot row is a reference, not a like-for-like comparison.

## Contamination risk
- The risk is high for any model trained on tasksource extraction ("Clairic-style", added in v2026.10.20; suite design decision L1). Its TAT-QA rows are train/dev contexts, and the build drops any test context they contain (see above). Whether Clairic or Sage 2 trained on tasksource is UNKNOWN.
- In general the risk is medium-high. The test gold has been on GitHub since Feb 2024 (the README says Jan 2024), and train/dev since 2021. Report this set as "old public" (suite design decision L18).

## Files
- `items.jsonl` has sha256 `49a6984ce5c7551fd110736e20173ff422867bf4394a7be35052f04caa3aee5e`, and `comparable/tatqa_gold_sample.json` has `feb86330…`. We checked that a rebuild is byte-identical.
- `build.py` re-creates `items.jsonl`, `build_report.json` and `comparable/tatqa_gold_sample.json` from `raw/` and checks the pinned sha256s. Run it with `/opt/homebrew/bin/uv run --no-project --with tiktoken python -I extract-bench/datasets/tatqa/build.py`.
- The scripts are `scripts/fetch_tasksource_tatqa.py` (tasksource rows), `scripts/selfcheck.py` (writes `selfcheck.out.json`) and `scripts/probe_parquet_footer.py`.
- The comparable files are `comparable/run_official.py` (the official scorer adapter, with `--selftest`), `comparable/official/` (the vendored scorer) and `comparable/selftest.out.json`.
- The paper copy is `refs/arxiv_2401.13223v3.pdf/.txt`.
- The build cost $0. No metered API was called.
- The planned run cost comes from `harness/estimate.py` at list prices, run locally only (`scripts/estimate.out.json`).
  - The totals are clairic_text $0.009 and haiku_json $0.015. The price for sage_precise and sage_fast is not public.
  - The Sage figure counts 131 calls because the estimator does not read `na_arms`. Skipping the 67 computed items leaves 64 calls. *(The estimator now honours `na_arms`, as of 2026-10-23.)* Set a $ ceiling before any of it runs.
- One provenance nit: in the tasksource fetch record, `bytes_read_cols` counts only row group 0. The real transfer was about 38 MB for validation plus about 90 MB per train file.

## Corrections (stated out loud)
- **Correction.** The spec says "Test, full 1,663 (https://arxiv.org/abs/2401.13223)". That paper's Table 1 gives 1,669 for TAT-QA test. 1,663 is the count in the released gold file. The published GPT-4 and GPT-3.5 rows use the paper's test set, so the 6-question difference is UNKNOWN.
- **Correction.** The claim "test gold public since Jan 2024" matches the README, but the commit that adds the file is dated 2024-02-14 (`12fd7c8b`).
- **Correction to the survey's `tatqa_text_overlap.out.json`.** Its 300-character prefix match put 246 train and 26 validation tasksource rows in "test_gold". With full states, 0 tasksource rows are TAT-QA test contexts. 1,487 are train, 188 are dev, and 20 match none. The prefix hits came from paragraphs that TAT-QA itself reuses across splits.

## Independent verification (2026-10-23, verifier agent; scripts in `/tmp/tatqa_vscripts/`, not kept)
- Running `build.py` on a copy reproduced `items.jsonl`, `build_report.json` and `comparable/tatqa_gold_sample.json` byte-for-byte (sha256 `49a6984c…`).
- For a gold spot-check, I drew 15 items with seed 7 and compared each against the raw test-gold record. I checked the question, answer, scale and expanded alternatives, and that every table cell and paragraph is present in `text`. There were 0 mismatches.
- All 150 items pass `harness.common.validate_item`. Ids are unique. Every acceptable gold scores 1.0 against itself. `comparable/tatqa_gold_sample.json` holds the same 150 uids.
- The git blob of `raw/tatqa_dataset_test_gold.json` is `3d5d3b69…`, which matches the GitHub contents API at 870accc4. That commit is HEAD (2024-12-09). Commit 12fd7c8b (2024-02-14) added the file. README line 105 gives CC BY 4.0, and the GitHub licence API says MIT. `raw/README.md` and `raw/leaderboard_index.html` are byte-identical to the pinned commit and the live leaderboard page.
- These baselines match the leaderboard: Human 84.1/90.8, TAT-LLM 70B 81.4/88.4, TAT-LLM 7B 76.4/85.1 and TagOp 50.1/58.0. GPT-4 71.92/79.71, GPT-3.5 59.47/68.11 and TAT-LLM 7B 74.56/82.88 match arXiv 2401.13223 Table 3 (test). The TagOp 58.0 F1 and human 90.8 F1 figures match the 2105.07624 abstract. I corrected the GPT setting and the TAT-LLM 7B note above.
- **The percent fraction alternative is lenient (still open and not fixed, because it is a scoring-policy question).** Percent arithmetic golds also accept `v/100`, and `numeric_tol` applies an absolute tolerance of 0.005 to it. That makes a ±0.5 percentage-point window. On the percent-number path the window is ±0.005.
  - Of the 28 percent items, a fraction rounded to 2 decimal places (which `decimals: 2` invites, for example 0.64 for 63.74%) gets credit on 28/28.
  - A fraction that is off by 0.4 percentage points gets credit on 11/28. The same error written as a percent gets credit on 0/28.
  - For 2 items (gold 0.35% and 0.1%), a prediction of `0` gets credit.
  - There are two fixes. Drop the `v/100` alternative, since the instructions already ask for the percent number, or make `numeric_tol` relative-only for that alternative. Either way, rerun `scripts/selfcheck.py` and `comparable/run_official.py --selftest` afterwards.
