# emails: Political fundraising emails (Willis LLM-Extraction-Challenge)

Built 2026-10-23 at a cost of $0. The build called no metered API. The suite plan is in the suite design notes, §2.10 and §8 row 10, with decisions L3, L4, L7, C7 and C11.

In brief:
- The set is 100 of 1,000 real political fundraising emails, sampled at random with seed 20. All of them are from 1 to 2 Nov 2024. There is one field, `committee`, which is the committee named in the "Paid for by" disclaimer.
- There are 0 null golds (0/100 and 0/1,000), so this set makes no abstention claim. An arm that returns null is wrong.
- The set has two scores, and they never appear in one table:
  - The repo matcher uses lower-cased exact match. It is the only score comparable to the published leaderboard.
  - The alnum-squashed score is fair to span extractors. The email bodies have line-join artefacts ("Harris forPresident"), so 6/100 golds (71/1,000) are not verbatim in the body. After squashing, 100/100 (1,000/1,000) are.
- We reproduce the published baselines exactly. Our re-implementation of `matcher.py`, run over the repo's own prediction files, matches all 8 checked rows of `summary_all_json.csv` (for example gpt-4o prompt2 0.969, 1,000 records).
  - On our 100 items, the same published outputs score 0.91 to 0.96, for example gpt-4o 0.94 [0.89, 0.98].
- In our judgement, n=100 with a ceiling of about 96% cannot separate systems closer than about 5 points, because the 95% CI half-width is about ±4 to 5 points. The 1,000-item anchor file is cheap to run (about 870k o200k tokens) and has more statistical power. Whoever sets the $ ceiling makes that call.

## 1. Source and revision

| What | Value |
|---|---|
| Repo | https://github.com/dwillis/LLM-Extraction-Challenge (GitHub API metadata in `raw/pages/repo.json`; 23 stars, 2 forks on 2026-10-23) |
| Commit | `72b8d5fe9a8b44702503005559438868bc73e000` (`main` HEAD, 2026-08-06, "Merge pull request #2 … smevals-integration"). The repo was created 2025-01-27 (`raw/pages/commits.json`) |
| Gold file | `fundraising-emails/training.csv`, git blob `a6dc4e52…`, sha256 `615db00f…59c2c4` (full hashes in `build_report.json`). Despite the name, this is the file `matcher.py` scores against. It last changed in commit `c181a6da75` (2025-07-10, "more evals") |
| Gold history | `raw/history/training_089dc69c.csv` is the 2025-01-27 version. A diff against HEAD shows that only 2 committee golds changed (rows 300 and 492, from "Harris forPresident" to "Harris for President"). Every other column is identical |
| Files used | `README.md`, `LICENSE`, and these under `fundraising-emails/`: `training.csv`, `summary_all_json.csv`, `matcher.py`, `email_llm.py`, `email_ollama.py`, `generate_leaderboard.py`, `make_matrix.py`, `index.html`, two `evals/classification_report_*` files plus `evals/model_performance_summary.csv`, `smevals-eval/*`, and 8 published prediction JSONs. That is 30 files (28 MB) in `raw/repo/`, fetched from `raw.githubusercontent.com/dwillis/LLM-Extraction-Challenge/<commit>/…` |
| Integrity | `build.py` checks the git blob sha1 of every `raw/repo` file against the commit tree (`raw/pages/tree.json`), and 30/30 match |
| Write-ups | Simon Willison's post of 2025-04-08 is at https://simonwillison.net/2025/Apr/8/political-email-extraction-leaderboard/ (saved as `raw/pages/willison_2025-04-08.html`). The repo's GitHub Pages leaderboard (https://dwillis.github.io/LLM-Extraction-Challenge/fundraising-emails/) 301-redirects to https://thescoop.org/LLM-Extraction-Challenge/fundraising-emails/, which returned HTTP 404 on 2026-10-23 |
| Downloaded code | We never executed downloaded code. We read `matcher.py`, the smevals checker and the prompts, and re-implemented them in our own scripts |

## 2. Licence

| Layer | Licence | Evidence |
|---|---|---|
| Repo (code, `training.csv` gold, model outputs) | MIT, "Copyright (c) 2025 Derek Willis" | GitHub API `license.spdx_id = MIT` (`raw/pages/license_api.json`); `raw/repo/LICENSE` |
| Email bodies | Political committees' mass fundraising emails, redistributed in the MIT repo. We did not check the original senders' terms | UNKNOWN beyond the repo's MIT grant |

- The bodies contain the dataset author's own subscription address and first name (for example "This email was sent to dpwillis67@…"). Both are already public in the repo, so sending them to an API adds nothing private.
- For redistribution status, see `LICENSE-DATA.md`.

## 3. Items and text

- Sampling works as follows:
  - `random.Random(20).sample(range(1000), 100)` gives the CSV row indices, which are written in row order. This is a simple random sample, because the spec asks for no stratification.
  - The item id is `emails-r<row:04d>`.
  - `meta.doc_id` is the item id, so the harness bootstrap clusters by email.
- As in the other sets (suite design decision L13), `random.Random(20).sample(sorted(ids), 20)` picks a holdout and sets `meta.holdout = true` on 20 items. The list is in `build_report.json`.
- The replication anchor, `anchor/items_all1000.jsonl`, holds all 1,000 rows in the same format (`meta.anchor_all1000 = true`). It is never mixed with the 100-item sample, and `list_datasets` does not see it.
- Item text:
  - `text` is the `body` column verbatim, so `text_source` is `native`.
  - The text includes no subject or headers. This matches prompt2, which sends only the body. Subject, from-address, date, domain and party are in `meta`.
  - The bodies are plain-text renderings of HTML emails and contain the repo's own line-join artefacts (lost spaces such as "careenough" and "asupporter"). We keep them as shipped.
- Of the bodies, 965/1,000 (96/100) contain the literal "paid for by". The remaining 35 phrase the disclaimer differently, for example "Paid for and authorized by" or "paid for and provided by" (`meta.paid_for_by_in_body`).
- Composition:
  - The sample has 70 distinct gold strings (69 after squash), and the full 1,000 have 260.
  - The largest clusters are NRSC (6 items) and Turning Point Action (4 items), and 49 committees appear once.
  - `meta.committee_cluster` (the squashed gold) supports an optional committee-clustered bootstrap.
  - By the repo's `party` column, the sample is D 57, R 36, blank 7.
- Token counts (o200k, tiktoken; `scripts/selfcheck.out.json`):

  | Set | p50 | p95 | max | total |
  |---|---:|---:|---:|---:|
  | sample (100) | 768 | 1,580 | 2,711 | 83,595 |
  | anchor (1,000) | 774 | 1,889 | 3,593 | 869,940 |

  - The number of items over Sage 2's 16,384 cap (content plus field definition) is 0.
  - Every item fits in one Clairic 8,192-token pass.
  - Sage's own tokenizer is UNKNOWN, so these are o200k estimates.
- Harness checks:
  - `harness/estimate.py --datasets emails` gives about $0.37 at list prices for all arms on the 100 items (`scripts/estimate.out.json`). This is an estimate, not a measurement.
  - A 5-item mock read and score (`dryrun-*`, no network, ledger unchanged) ran clean, and we then deleted it.

## 4. Field

| Field | Item type | Clairic | Sage 2 | Gold | Null golds |
|---|---|---|---|---|---:|
| committee | text | text (span) | text | the `training.csv` committee string, verbatim (no variants added) | 0 |

- `instructions` is the committee clause of the repo's prompt2, verbatim: *The name of the committee in the disclaimer that begins with "Paid for by" but do not include the "Paid for by" text itself. If no committee is present, use null.*
- The repo also has a sender gold (`name`). The leaderboard does not score it, so it is not a field here. The build keeps it as `meta.repo_sender_unscored`.
- `meta.gold_reachable.committee` records whether the gold is in the body verbatim (whitespace-collapsed, lower-cased). `meta.gold_reachable_squashed` is the same test after alnum-squash. `meta.gold_body_span` is the body substring (with offsets) whose squash equals the gold's squash.
- The 6 non-verbatim sample items are all lost-space artefacts. The table gives the body text and the gold for each:

  | Item | Body text | Gold |
  |---|---|---|
  | r0023 | "Hung Cao forVirginia" | "Hung Cao for Virginia" |
  | r0280 | "DavidPan for Congress 2024" | "David Pan for Congress 2024" |
  | r0416 | "INDIANAREPUBLICAN STATE COMMITTEE" | "INDIANA REPUBLICAN STATE COMMITTEE" |
  | r0926 | "Josh Hawley VictoryCommittee" | "Josh Hawley Victory Committee" |
  | r0962 | "Bernie Moreno forSenate" | "Bernie Moreno for Senate" |
  | r0979 | "STEVE GARVEYFOR U.S. SENATE" | "STEVE GARVEY FOR U.S. SENATE" |

## 5. Metric and why

`metric: exact_norm` is the harness default. The scorer for all three views below is `comparable/score_emails.py`, which takes `ITEMS.jsonl` and a `PREDS.jsonl` of `{id, committee}`.

| View | Rule | Use |
|---|---|---|
| repo_matcher (COMPARABLE) | The `matcher.py` Accuracy line, an exact match after `fillna('none').astype(str).lower()`. The Accuracy line does not strip, and a null prediction becomes `"none"` | The only number that may sit next to the leaderboard, and only for a prompt2-style run (for example the Haiku/OpenRouter control given `comparable/prompt2.txt` and `prompt2_schema.json` verbatim) |
| exact_norm (harness) | NFKC, casefold, whitespace collapsed, outer punctuation stripped | Clairic versus Sage head-to-head with the suite's other text fields |
| squashed (FAIR) | Alphanumerics only (`harness/score.py` `squash`) | Removes the 6/100 lost-space handicap for span extractors (Clairic and Sage return spans of the body). It also forgives the inconsistent gold spacing in §8 |

- There are two headline views for a reason. Clairic and Sage can only return text that is in the body, so under the repo matcher they lose up to 6 points on this sample through no fault of their own. An LLM rewrites the spacing, and published gpt-4o got all 6 of these items right. The repo matcher stays the only like-for-like score against the public rows.
- `score_emails.py` also reports repo_matcher accuracy split into verbatim (94) and non-verbatim (6) items, per suite design decision C9.

## 6. Published baselines

The repo author scored every row with `matcher.py`, as committee accuracy on the same 1,000 emails (`training.csv` at the pinned commit). The prompt is prompt2 unless noted. "Total Records" is the number of rows `matcher.py` merged, and it is 1,000 for every row below.

| System | Score | Setting | URL |
|---|---:|---|---|
| gpt-4o (`gpt-4o_november_2024_prompt2.json`) | 0.969 | all 1,000, prompt2, repo matcher | https://github.com/dwillis/LLM-Extraction-Challenge/blob/72b8d5fe9a8b44702503005559438868bc73e000/fundraising-emails/summary_all_json.csv |
| gpt-4o-mini (`gpt-4o-mini_november_2024.json`, not a prompt2 file) | 0.969 | all 1,000, prompt version unlabelled | same |
| claude-3.7-sonnet | 0.964 | all 1,000, prompt2 | same |
| gemini-2.5-pro (`gemini_25_…_prompt2`) | 0.962 | all 1,000, prompt2 | same |
| claude-3.5-haiku | 0.956 | all 1,000, prompt2 | same |
| gpt-4.1-mini | 0.932 | all 1,000, prompt2 | same |
| claude-haiku-4.5 | 0.929 | all 1,000, prompt2 | same |
| gpt-4.1-nano | 0.884 | all 1,000, prompt2 | same |

- `scripts/replicate_published.py` writes `scripts/replicate_published.out.json`. Our re-implementation of `matcher.py`, applied to the repo's own prediction JSONs, reproduces all 8 rows exactly (accuracy and record count).
- The CSV has 79 rows in all. 56 are prompt2 runs, and 63 merged all 1,000 records. Notable prompt2 rows include claude-sonnet-4.5 at 0.768, gpt-5-mini at 0.774 and gpt-oss-120b at 0.771. Exact-match scoring punishes extra text (addresses, "Inc." handling and similar). Whether that explains each model's score is UNKNOWN, and we did not check it.
- The next table restricts the same 8 published outputs to our 100 items. We computed these numbers from the published predictions, so they are not published numbers. The CIs come from an item-level bootstrap (seed 20, 10,000 resamples). The sources are `scripts/replicate_published.out.json` and `scripts/score_published_sample.out.txt`.

  | System | repo_matcher | squashed |
  |---|---:|---:|
  | claude-3.7-sonnet | 0.96 | 0.98 |
  | gemini-2.5-pro | 0.96 | 0.98 |
  | gpt-4o | 0.94 [0.89, 0.98] | 0.96 [0.92, 0.99] |
  | gpt-4o-mini | 0.94 | 0.96 |
  | claude-3.5-haiku | 0.94 | 0.96 |
  | gpt-4.1-mini | 0.93 | 0.97 |
  | claude-haiku-4.5 | 0.91 [0.85, 0.96] | 0.93 [0.88, 0.98] |
  | gpt-4.1-nano | 0.91 | 0.93 |

  - The committee-clustered CIs (`scripts/cluster_ci.out.txt`) are gpt-4o [0.879, 0.989] and claude-haiku-4.5 [0.824, 0.979].
  - The per-item outputs are in `comparable/published_predictions.jsonl` (1,000 × 8) and `comparable/preds_<model>.sample.jsonl`.
- Some historical numbers are not comparable:
  - The Willison post (2025-04-08) states "Gemini 2.5 Pro tops the leaderboard at the moment with 95.40%" and that Mistral Small 3.1 "manages 5th place with 85.70%".
  - Those numbers predate the "better matching" commit `b41d0e4` (2025-04-10) and the gold fix `c181a6d` (2025-07-10). The current CSV has gemini-2.5-pro at 0.962.
  - The committed `index.html` is also stale against the CSV. It shows claude-haiku-4.5 prompt2 with 811 records at 92.73%. Use the CSV.
- Prompt provenance:
  - The smevals README in the repo (commit `60400b7`, 2026-08-06) says the prompt in `email_llm.py` is "the 'prompt2' text" and that its checker is comparable to the Accuracy column "for `*_prompt2.json` runs only".
  - `email_llm.py` itself was first committed on 2025-12-06, but the first prompt2 files date from 2025-02-24. Whether every older prompt2 run used exactly this string and schema is therefore UNKNOWN.
  - The runs used Simon Willison's `llm` library with a JSON schema. Temperature and other decoding settings are UNKNOWN.

## 7. Replication anchor (suite design decision L3; plan only, needs a $ budget)

- Run gpt-4o with `comparable/prompt2.txt` (the email body substituted for `{body}`) and `comparable/prompt2_schema.json` over `anchor/items_all1000.jsonl`. Score with `score_emails.py` (`repo_matcher`) and compare to 0.969.
- The plan had a harness gap. The harness OpenRouter arm uses its own system prompt (`clients.OR_SYSTEM`) plus a per-field schema. It has no per-dataset prompt override, so a prompt2-verbatim arm needs a small harness addition. This builder did not edit the harness. *(This was resolved on 2026-10-23. OpenRouter `#comparable` arms now send `meta.comparable.prompt` and the schema verbatim, with no harness system prompt, through `clients.openrouter_native_body`. One example is `or:openai/gpt-4o:text#comparable` on `emails~anchor`.)*
- In our judgement, scaling the 100-item `estimate.py` numbers to 1,000 items gives the following at list prices. Clairic on all 1,000 costs about $0.11 (about 10.4 times the tokens). Sage 2 makes 10 times the calls, and its extraction price is not public yet. At p of about 0.95, the 95% CI half-width falls from about ±4.3 points (n=100) to about ±1.4 points (n=1,000). The full run is worth it if the cap allows.

## 8. Gold quality (suite design decision L7)

- On 2/1,000 items, all 8 published models agree with each other but not with the gold (1 of them is in the sample, r0239). In both, the gold is "Pennsylvania Democratic Party" and the models say "The Pennsylvania Democratic Party". The body reads "Paid for by The Pennsylvania Democratic Party." (`scripts/gold_disagreement.out.json`). This is a candidate for blind review. We have not changed the gold.
- The sample shows inconsistent gold conventions. The squash view fixes the spacing cases but not the article or suffix cases:
  - Some golds keep a lost-space artefact: "ShontelBrown for Congress" (r0224) and "DefeatBoebert PAC" (r0343). Here the gold copies the artefact, but in the 6 items of §4 the gold repairs it.
  - The gold "THE REPUBLICAN PARTY OF WISCONSIN" (r0415) keeps the leading article "THE", while "Pennsylvania Democratic Party" drops it.
  - The gold "Keith Ellison for Attorney General committee" (r0262, r0818) keeps the trailing word "committee".
- At least one of the 8 published models misses 18/100 sample items under the repo matcher, and 13/100 under squash.

## 9. Contamination and legitimacy

- The contamination risk is medium.
  - The gold and 79 model-output files have been public on GitHub since 2025-01-27 (`training.csv` is in the initial commit). The smevals packaging (2026-08) makes the set easier to reuse as an eval.
  - The emails themselves are from Nov 2024.
  - The suite design survey (decision L1) found no fundraising-emails source among the 127 train sources of the tasksource extraction corpus (checked by name only).
- On legitimacy, the set is independent. It has a single author, Derek Willis (University of Maryland journalism), and Simon Willison has covered it. It is not peer reviewed. It has a single field, with a ceiling of about 97%.
- These are real marketing emails, not support tickets. Support tickets and claims remain a declared gap (suite design decision C11).

## 10. Files

| Path | What |
|---|---|
| `items.jsonl` | 100 items (common format) |
| `anchor/items_all1000.jsonl` | All 1,000, for the replication anchor only |
| `build.py` | Rebuilds both from `raw/` (stdlib plus optional tiktoken) with `uv run --no-project --with tiktoken python -I build.py` |
| `build_report.json` | Counts, hashes, token stats, sample rows, holdout ids, non-verbatim list |
| `comparable/` | `score_emails.py` (three views), `prompt2.txt`, `prompt2_schema.json`, published predictions |
| `scripts/` | `selfcheck.py`, `replicate_published.py`, `export_published_preds.py`, `gold_disagreement.py`, `cluster_ci.py`, each with its `*.out.*` output, plus `estimate.out.json` |
| `raw/repo/` | Pinned repo files (blob-verified) |
| `raw/history/` | The 2025-01-27 `training.csv` |
| `raw/pages/` | GitHub API JSON (repo, licence, tree, commits, file histories), the Willison post and the 404 check |
