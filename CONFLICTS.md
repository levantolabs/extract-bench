# Conflicts of interest

Levanto Labs builds and maintains Extract Bench, and Levanto also sells Sage 2, one of the systems on the leaderboard.
A benchmark that a vendor builds tends to rank that vendor highly. Read our numbers with that in mind, and check them:
this repo has every prediction and the scorer.

## What we do about it

| Mitigation | Status in v0.1 |
|---|---|
| The leaderboard, the figures and `metadata.json` (`maintainer_product: true`) mark the maintainer's row with \* | Done |
| We publish all predictions for every system, and one public command (`extract-bench report`) re-scores them | Done |
| Every system, Sage 2 included, runs through a public adapter in this repo | Done |
| The per-dataset board and the Sage 2 vs Clairic chart show where Sage 2 is behind | Done |
| The same tuning budget for every system | Not yet. Only Sage 2's abstention thresholds were tuned, on the `fit` half. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md) |
| Item selection that doesn't depend on one system | Not yet. Sage 2 defines the v0.1 single-request filter |
| Other vendors review their own adapter configuration | Not yet |
| An independent party re-runs the maintainer's row and the top rows | Not yet |
| No paid placement, and no published row is ever removed | Policy |

If you represent a listed system and think its adapter or configuration is unfair, open an issue. We'll fix it in the
open, re-run the system, and record the change in [CHANGELOG.md](CHANGELOG.md).
