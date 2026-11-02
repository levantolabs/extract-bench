# squad2: SQuAD 2.0 dev minus tasksource eval overlap (Tier C, 60 answerable and 60 unanswerable)

We built this set on 2026-10-23 at $0 and called no metered API. The suite plan is in the suite design notes (§2.9, §8 row 12, and suite design decisions L1, L17, C2 and C12).

- The set has 120 items: 60 answerable and 60 unanswerable. Each item is one question about one Wikipedia paragraph from the SQuAD 2.0 dev split.
  - Every item uses a different paragraph, and the items cover all 35 dev articles, with 1 to 4 items each.
  - The gold is the list of human reference spans. For an unanswerable question, the gold is `null`.
- **Tier C.** We report it apart from the headline, for two reasons:
  - it is old, heavily contaminated public data;
  - it is QA-shaped rather than a business-document task.
- The set tests abstention over a passage. When the answer is not in the text, does the extractor return null (Sage 2) or guess a span (Clairic)? An earlier web-search experiment (21 Oct 2026) showed this trade-off. Clairic answered 97 to 100% of questions, and Sage 2 answered 29 to 72%.
- The tasksource exclusion is stricter than the spec. We drop every dev paragraph that appears anywhere in the tasksource extraction validation/test splits, whatever the source. That removes 871 of the 1,204 paragraphs and leaves 333 eligible (§3).
- This corrects the suite spec. Its "1,000 contexts" were 1,000 tasksource *rows*, which cover only 791 distinct dev paragraphs:
  - 191 paragraphs are split across a validation row *and* a test row.
  - A further 80 dev paragraphs sit in *other* tasksource sources: AdversarialQA, MuSiQue, ConcurrentQA and QA-SRL. The spec did not know about these.
- This also corrects the survey, which said there was "no verified LLM zero-shot row". GPT-3 and PaLM both publish zero-shot SQuAD 2.0 dev F1 (§6). Those numbers are on the full dev set, not on our subset, so they are reference points only.
- The pre-registered headline rules are (a) null, and (b) null or confidence below 0.5. We show the official `best_f1` only as an oracle-threshold upper bound (suite design decision L17).

## 1. Source and revision

| What | Value |
|---|---|
| Benchmark page | https://rajpurkar.github.io/SQuAD-explorer/ (saved as `raw/pages/squad_explorer_index.html` and `.txt`) |
| Primary file | `raw/official/dev-v2.0.json`, from https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json. It has sha256 `80a5225e…182d8` and git blob `c6fa1a89a12c90b7fbfa8f6fb437cfff27af41ca` |
| Repo revision | GitHub `rajpurkar/SQuAD-explorer` master is `eee5fdbf62f8613a7812b03419e6b29617b74fd1` (2023-10-12, "Update 2.0"). The blob of `dataset/dev-v2.0.json` in that tree equals our file. The file was last changed in `240e165a` (2018-07-02, "add dataset files"). See `raw/pages/gh_squad_explorer_*.json` |
| Official scorer | `evaluate-v2.0.py`. The site links it on worksheets.codalab.org, whose TLS certificate had expired on 23 Oct, so we took the copy from the same repo (blob `21f52c179c8c…`, added in `09eac997`, 2021-05-27). We saved it as `raw/official/evaluate-v2.0.py.txt` and do not execute it |
| HF mirror (cross-check only) | `rajpurkar/squad_v2` at `3ffb306f725f7d2ce8394bc1873b24868140c412`, file `squad_v2/validation-00000-of-00001.parquet`. The LFS sha256 `0560174a…24cb` matches. All 11,873 rows match the official JSON in id, title, context, question, answer texts and starts (`hf_crosscheck` in `build_report.json`). This is also the revision tasksource used (its `build-manifest.json` has `"squad-v2": "3ffb306f…"`) |
| Paper | Rajpurkar, Jia and Liang, "Know What You Don't Know: Unanswerable Questions for SQuAD", ACL 2018, https://arxiv.org/abs/1806.03822 (PDF and text in `raw/pages/`) |

We counted the dev split from the file:

| Articles | Paragraphs | Questions | Answerable | Unanswerable |
|---:|---:|---:|---:|---:|
| 35 | 1,204 | 11,873 | 5,928 | 5,945 |

Each unanswerable question carries the crowdworker's "plausible" (wrong) answer. The test set is hidden, and the site says "we do not release the test set to the public".

## 2. Licence

| Layer | Licence | Evidence | Status |
|---|---|---|---|
| SQuAD 2.0 data | CC BY-SA 4.0 | The site says "Download a copy of the dataset (distributed under the CC BY-SA 4.0 license)" (`raw/pages/squad_explorer_index.txt`). The HF card says `license: cc-by-sa-4.0` (`raw/pages/hf_squad_v2_README.md`, API `cardData.license`) | Verified |
| SQuAD-explorer repo code (including the scorer) | MIT | GitHub API `license.spdx_id = MIT` (`raw/pages/gh_squad_explorer_repo.json`) | Verified |
| AdversarialQA (used only for the overlap check) | CC BY-SA 4.0 | HF API `cardData.license` (`raw/pages/hf_adversarial_qa_api.json`) | Verified |
| tasksource extraction (used only for the overlap check) | `other` (repo-level). We did not read the per-row `license` column | HF API `cardData.license` | Partial |

- CC BY-SA requires attribution and share-alike for any redistribution.
- For redistribution status, see `LICENSE-DATA.md`.

## 3. Tasksource exclusion (suite design decision L1)

The corpus is `tasksource/tasksource-jev-typed-decisions`, config `extraction`.
- Its revision is main `8b1afcc6ad12554b347134a91b5f0dc6ae2361db` (2026-10-21T23:44Z, tag `v2026.10.22`).
- Its README describes the config as "one text and several typed fields to fill from it, answered in one pass, as extraction models (e.g. Cloudflare's Clairic) are asked" (`raw/tasksource/extraction_README.md`).
- Its HF tags include `jev`, `clairic` and `structured-extraction` (`raw/pages/hf_tasksource_api.json`).
- We downloaded the validation (62.7 MB) and test (59.6 MB) parquet files. Their LFS sha256 values match `raw/pages/hf_tasksource_paths_info.json`.
- We did not download the 780 MB train parquet, because HF throughput was 25 to 250 KB/s on 23 Oct.

A dev paragraph is ineligible if any of these rules holds. The full text is in the `build.py` docstring.

| Rule | Test | Paragraphs hit |
|---|---|---:|
| a | One of its question ids is in a tasksource `squad-v2` row's `metadata.question_ids`. Validation and test rows come from the parquet, and train rows come from the survey pull of the same revision. Train holds 0 dev ids | 791 |
| b | Its full text equals, contains or is contained in a `squad-v2` validation/test state, or its first 120 characters match a `squad-v2` state or head in any split. The train heads come from the survey pull | 791 |
| c | An 80-character probe (start, middle or end, with whitespace collapsed) occurs in any validation/test row of any source. The search runs Aho-Corasick over 44,205 plus 45,112 rows (`scripts/xsource_probe.py` writes `derived/xsource_hits.json`) | 871 |
| | **Union (excluded)** | **871 paragraphs (8,676 questions)** |
| | **Eligible** | **333 paragraphs in all 35 articles (1,586 answerable and 1,611 unanswerable questions)** |

Rule c also found paragraphs beyond squad-v2. The table counts distinct dev paragraphs per source and split, and a paragraph can appear in several sources.

| Split | Source | Dev paragraphs |
|---|---|---:|
| test | adversarial-qa | 162 |
| test | concurrentqa | 1 |
| test | qasrl/orig | 1 (one sentence of `Steam_engine#9`) |
| validation | adversarial-qa | 160 |
| validation | musique | 77 (e.g. `Southern_California#4`, which appears in full inside a 9,450-character MuSiQue state) |

- 79 paragraphs are in other sources only, and none of their questions are in the squad-v2 rows.
- One more paragraph (`Economic_inequality#20`) repeats a sentence of `#29`, which is in a squad-v2 row.

AdversarialQA matters because the original `UCLNLP/adversarial_qa` at `c2d5f738` (the revision in tasksource's manifest) reuses SQuAD dev paragraphs in its validation and test passages (`scripts/advqa_overlap.py` writes `derived/advqa_overlap.out.json`):

| AdversarialQA split | Its passages | SQuAD 2.0 dev paragraphs among them (exact / 120-char head) |
|---|---:|---:|
| validation | 416 | 160 / 164 |
| test | 409 | 285 / 287 |
| train | 2,648 | 0 / 0 |

Two consistency checks pass:
- The tasksource eval question ids in the survey pull (refs/convert/parquet `8614e633`) and in the main parquet are identical (4,789 = 4,789).
- All 98/98 rows we fetched from the datasets-server equal the main-parquet rows (`derived/convert_vs_main.out.json`).

Train-side coverage is partial. It covers:
- squad-v2 train question ids (0 dev);
- squad-v2 train 120-character heads (1 hit, `Immune_system#39`, which rule a already excludes);
- AdversarialQA's original train split (0 dev paragraphs).

We did not scan the other train sources of the tasksource corpus, so their overlap is UNKNOWN. The per-paragraph reasons are in `derived/exclusion.json`.

## 4. Sampling (seed 20, pre-registered in `build.py` before the first run)

1. Take the eligible paragraphs. Sort the articles by title and shuffle them with `random.Random(20)`. Within each article, sort the paragraphs by index and shuffle them.
2. Go round-robin over the articles, one paragraph per visit. Each paragraph gives one question, from the stratum with fewer picks so far (ties go to answerable). If the paragraph has no question of that stratum, use the other stratum. Pick the question with `rng.choice` over its sorted ids. Stop at 60 answerable and 60 unanswerable.
3. For the sealed holdout, a fresh `random.Random(20).sample(sorted(ids), 12)` per stratum gives 24 items (12 and 12) with `meta.holdout = true`.
4. Rebuilds are deterministic. Two runs gave the same items.jsonl sha256, `2496506c…3628dc`.

The 50/50 split matches the dev split's own mix (5,928 / 5,945) and the eligible pool's (1,586 / 1,611). It does not match business documents, where most asked fields are present. In our judgement, this is one more reason the set is Tier C.

## 5. Items and fields

| Key | Value |
|---|---|
| `id` | `squad2-<SQuAD question id>` |
| `text` | The paragraph, verbatim, with `text_source: native`, `modality: text` and `images: []` |
| `fields` | One field: `{"name": "answer", "type": "text", "instructions": <the question, verbatim>, "gold": [...] \| null}` |
| `gold` (answerable) | De-duplicated reference answer texts whose SQuAD normalisation is non-empty, the same filter the official scorer uses. 23 items have 1 distinct gold, 29 have 2, 7 have 3 and 1 has 4 |
| `gold` (unanswerable) | `null` (60 items). The crowdworker's plausible wrong answer is in `meta.plausible_answers`, and 60/60 items have one |
| `metric` | `token_f1` |
| `meta` | `split`, `squad_version`, `qid`, `article`, `paragraph_index`, `doc_id` (the bootstrap cluster, with one item per paragraph), `is_impossible`, `stratum`, `answers` (all references with `answer_start`, duplicates kept as in the official file), `plausible_answers`, `n_questions_in_context`, `question_raw`, `sage_shape`, `sage_question_needs_sanitising`, `holdout`, `tokens_o200k` |

We declared these call shapes (suite design decision C2):

| Arm | Call |
|---|---|
| Clairic | `fields: {"answer": {"type": "text", "instructions": <question>}}`, with `state` set to the paragraph. Its text answers are spans with offsets, which is the SQuAD task shape |
| Sage 2 | Bare-question mode, with `extract` set to the question string (`harness/clients.py` `sage_question`, at most 256 characters, with quotes replaced by `'`). All questions are at most 150 characters, and 1 contains a double quote |
| Type map | `text` maps to `text` for both, and there are no list or number fields |

- All 255 reference and plausible spans sit at their `answer_start` in `text` (255/255), so every span is reachable.

## 6. Metrics

### TYPED view: the head-to-head (harness `score.py`)

- `token_f1` uses SQuAD v2 normalisation (lower-case, strip punctuation and articles) and takes the max over the gold list.
- The two rules decide what counts as an abstention:
  - under rule (a), the value is null;
  - under rule (b), the value is null or its confidence is below 0.5.
- Scoring works like this:
  - an abstention on a null gold scores 1 (`correct_abstain`);
  - an answer on a null gold scores 0 (`hallucinated`);
  - an abstention on an answerable question scores 0 (`false_abstain`).
- This matches the official per-question rule at a fixed threshold, where an unanswerable question scores 1 only for the empty answer.
- Report these:
  - overall, HasAns and NoAns separately;
  - the abstention table under both rules;
  - Clairic also at its default `null_threshold` of 0.5, as in suite design decision C7.

### COMPARABLE view: `comparable/squad_v2_eval.py`

- This is a line-by-line re-implementation of `evaluate-v2.0.py`, which we read but did not run, restricted to our 120 items.
- It outputs `exact`, `f1`, `HasAns_*`, `NoAns_*` and the oracle `best_exact` / `best_f1`.
- The mapping to `na_prob` works like this:
  - Clairic `p_null` maps directly.
  - For Sage 2 the mapping is an assumption, UNKNOWN until the new spec. It is 1 minus the `probability` of the returned value, or 1.0 when Sage returns null.
- best_f1 is not a headline (suite design decision L17), because it picks the threshold on the same items it scores. The self-test shows how optimistic that is. If every unanswerable question gets its plausible wrong answer at confidence 0.9, `f1` is 50.0 but `best_f1` is 100.0, because the oracle threshold sits exactly at the tie.
- The official `find_best_thresh` walks the questions in a stable sort by `na_prob`, so tied probabilities make `best_*` depend on item order. We keep the items.jsonl order.

### Self-check (`scripts/selfcheck.py` writes `scripts/selfcheck.out.json`)

| Check | Result |
|---|---|
| Lines parse; harness `validate_item` passes; field name, type, metric and modality are valid; the `gold` key is present | 120/120 |
| Gold is null exactly when `is_impossible` is true; one item per paragraph; ids are unique | Pass |
| Every gold value scored as a prediction (TYPED, rules a/b) | 212/212 `answered` = 1.0 |
| Null on null gold | 120/120 `correct_abstain` |
| Plausible answer on null gold | `hallucinated` |
| Answer with confidence 0.3 on null gold | `hallucinated` under (a), `correct_abstain` under (b) |
| COMPARABLE, gold as predictions | EM 100 / F1 100 |
| COMPARABLE, all null | Overall 50, HasAns 0, NoAns 100 |

## 7. Published baselines

These rows are reference points and are not paired with our items. Every row is on the full dev set or the hidden test set, not on our 120-item tasksource-clean subset. Only our own arms (Clairic, Sage 2, Haiku 5.5 JSON control) are paired.

| System | Score | Setting | Split | URL |
|---|---|---|---|---|
| Human performance | 86.831 EM / 89.452 F1 | Leaderboard row "Stanford University (Rajpurkar & Jia et al. '18)" | test (hidden) | https://rajpurkar.github.io/SQuAD-explorer/ |
| IE-Net (ensemble), RICOH_SRCB_DML, rank 1 | 90.939 EM / 93.214 F1 | Fine-tuned on SQuAD 2.0 train (leaderboard submission) | test | https://rajpurkar.github.io/SQuAD-explorer/ |
| Human | 86.3 EM / 89.0 F1 (dev); 86.9 / 89.5 (test) | Table 3 | dev / test | https://arxiv.org/abs/1806.03822 |
| DocQA + ELMo | 65.1 / 67.6 (dev); 63.4 / 66.3 (test) | Fine-tuned, 2018 (Table 3) | dev / test | https://arxiv.org/abs/1806.03822 |
| DocQA | 61.9 / 64.8 (dev); 59.3 / 62.3 (test) | Table 3 | dev / test | https://arxiv.org/abs/1806.03822 |
| BNA (BiDAF-No-Answer) | 59.8 / 62.6 (dev); 59.2 / 62.1 (test) | Table 3 | dev / test | https://arxiv.org/abs/1806.03822 |
| GPT-3 175B, zero-shot | 59.5 F1 / 52.6 EM | Prompted LM, in-context (Table 3.7 and the Appendix H full-results table) | dev (11,873) | https://arxiv.org/abs/2005.14165 |
| GPT-3 175B, one-shot / few-shot (K=16) | 65.4 / 69.8 F1; 60.1 / 64.9 EM | same | dev | https://arxiv.org/abs/2005.14165 |
| "Fine-tuned SOTA" as cited by GPT-3 | 93.0 F1 / 90.7 EM | Cited from [QIA20] | dev (per its table) | https://arxiv.org/abs/2005.14165 |
| PaLM 540B, 0-shot | 80.8 F1 / 75.5 EM | Prompted LM (Table 4) | "same splits as … Brown et al. (2020)", i.e. dev | https://arxiv.org/abs/2204.02311 |
| PaLM 540B, 1-shot / few-shot (5) | 82.9 / 83.3 F1; 78.7 / 79.6 EM | Table 4 | dev | https://arxiv.org/abs/2204.02311 |
| Prior SOTA cited by PaLM (GLaM 62B/64E), 0-shot | 71.1 F1 / 64.7 EM | Table 4 | dev | https://arxiv.org/abs/2204.02311 |

- The LLM rows use generative prompting with a free-text answer, not a span-extraction API with null.
- The prompt formats differ from ours.
- The GPT-3 and PaLM rows come from base models of about 2020-22. We found no current frontier zero-shot row. In our judgement, Haiku 5.5 JSON is the relevant control.

## 8. Contamination risk: high

- **Public exposure.** The dev gold has been public since at least 2018-07-02, when it entered the repo history. The paragraphs are Wikipedia text from SQuAD 1.1 (2016).
- GPT-3 flagged 11,136 / 11,873 (94%) of SQuADv2 dev examples as potentially contaminated. Its paper says "the source text was present in our training data but the question/answer pairs were not" (§4 and its clean-subset table, https://arxiv.org/abs/2005.14165).
- PaLM lists SQuADv2 under "Wholesale contamination". Its Table 18 gives a clean proportion of 14.8% (https://arxiv.org/abs/2204.02311).
- The tasksource extraction corpus, whose tags include `clairic`, holds SQuAD 2.0 train in its train split. It also held 871 of the 1,204 dev paragraphs in its eval splits, and we removed all of them.
- Two things are not checked. Whether Clairic or Sage trained on tasksource is UNKNOWN, and we did not scan the corpus's other train sources (§3).
- Read results as "old public set" in the suite's old-vs-fresh split.

## 9. Size, tokens and cost

Token counts (o200k, tiktoken; `scripts/selfcheck.out.json`) are:

| What | p50 | p95 | max | total |
|---|---:|---:|---:|---:|
| Paragraph text | 141 | 278 | 705 | 19,413 |
| Text + question | 151 | 290 | 713 | 20,839 |

- No item comes near Sage 2's 16,384 cap or Clairic's 8,192-token pass, so there is no windowing.

The list-price estimate below comes from `harness/estimate.py --datasets squad2`, with output in `scripts/estimate.out.txt`. No arm was run.

| Arm | Estimate |
|---|---|
| clairic_text | about $0.004 |
| sage_precise / sage_fast | price not public |
| haiku_json | about $0.008 |
| **All arms** | **about $0.41** |

On disk, `raw/` is 148 MB: tasksource eval parquet 124 MB, pages and PDFs 12 MB, official 4 MB, AdversarialQA 6 MB and HF 1.3 MB.

## 10. Risks and limits

- **Wide CIs.** n = 120 (60/60), and HasAns and NoAns each have n = 60. In our judgement, that means plus or minus 10 to 12 points per stratum.
- **Oversampled abstention.** At 50% unanswerable, the set has far more unanswerable questions than business use, which is why it is Tier C and outside the headline.
- **Not a business use case.** It is Wikipedia QA. It measures passage-level null behaviour, not typed-field extraction.
- **The question is the field instruction.** We ask Clairic for one field whose instructions are a question. That is a legitimate Clairic call shape, but not its typical one, which uses named schema fields.
- **Exclusion bias.** We removed 72% of paragraphs (871/1,204), but the pool looks similar on the surface, and all 35 articles are still represented. `scripts/pool_compare.py` wrote `derived/pool_compare.out.json`:

  | Pool | Mean questions per paragraph | Median length (chars) | Unanswerable share |
  |---|---:|---:|---:|
  | Eligible | 9.60 | 700 | 0.506 |
  | Excluded | 9.96 | 712 | 0.502 |

  The effect on difficulty is UNKNOWN.
- We assume the Sage `na_prob` semantics (§6). They matter only for the COMPARABLE oracle view, not for rules (a) and (b).

## 11. Files and rebuild

```sh
cd ./datasets/squad2
U=/opt/homebrew/bin/uv
$U run --no-project --with pyarrow python -I scripts/advqa_overlap.py "$PWD"                        # derived/advqa_overlap.out.json
$U run --no-project --with pyarrow --with pyahocorasick python -I scripts/xsource_probe.py "$PWD"   # derived/xsource_hits.json
$U run --no-project --with pyarrow python -I scripts/convert_vs_main.py "$PWD"                     # derived/convert_vs_main.out.json
$U run --no-project --with tiktoken --with pyarrow python -I build.py                              # items.jsonl, build_report.json, derived/exclusion.json
$U run --no-project --with tiktoken python -I scripts/selfcheck.py                                 # scripts/selfcheck.out.json
$U run --no-project python -I scripts/pool_compare.py "$PWD"                                        # derived/pool_compare.out.json
```

| Path | What |
|---|---|
| `raw/official/` | `dev-v2.0.json`, `evaluate-v2.0.py.txt` (not executed) |
| `raw/hf/` | HF validation parquet (cross-check) |
| `raw/tasksource/` | Extraction validation/test parquet at 8b1afcc6, `build-manifest.json`, `extraction_README.md`, the survey's squad/tatqa row pull (train heads), and 98 datasets-server rows |
| `raw/adversarial_qa/` | Original AdversarialQA parquet at c2d5f738 |
| `raw/pages/` | Explorer page, HF/GitHub API metadata, arXiv PDFs and text (1806.03822, 2005.14165, 2204.02311) |
| `scripts/` | `fetch_tasksource.py` (datasets-server fetch; we abandoned its duckdb variant as too slow), `xsource_probe.py`, `advqa_overlap.py`, `convert_vs_main.py`, `selfcheck.py`, `pool_compare.py`, `survey_ts_parquet.py.txt` (the survey's original pull script, for provenance) |
| `comparable/squad_v2_eval.py` | Official-logic EM/F1 and the na_prob oracle |
| `derived/` | Overlap and exclusion outputs |
