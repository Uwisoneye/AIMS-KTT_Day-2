# process_log.md

## Timeline

### Hour 1
- Read the challenge brief end-to-end.
- Chose a CPU-safe baseline: logistic regression for outage probability and ridge regression for outage duration.
- Recreated the synthetic inputs because only the brief was available in this workspace.
- Defined the appliance catalog and three business archetypes.

### Hour 2
- Built lagged 24h-ahead features.
- Implemented walk-forward evaluation over the last 30 days.
- Tuned the outage generator to match the brief more closely: around 4% outage rate and about 90 minutes average outage duration.

### Hour 3
- Implemented the prioritization rule.
- Added a risk-adjusted power cap and category-based drop order.
- Built the static `lite_ui.html` and exported per-business plans.

### Hour 4
- Wrote `digest_spec.md`.
- Added README, notebook, metrics artifacts, and signed-file placeholders.
- Reviewed outputs and prepared the repo for submission.

## Tools used

### ChatGPT
- Purpose: reasoning support, architecture review, README drafting, and code generation assistance.
- Why used: to move faster under the 4-hour constraint while keeping the solution explainable.

### Python + pandas + scikit-learn
- Purpose: data generation, feature engineering, training, evaluation, and artifact export.

## Three sample prompts used

1. "Help me turn this outage challenge brief into the smallest end-to-end repo that still scores well on technical quality and product adaptation."
2. "Draft a simple prioritizer function that enforces drop luxury before critical and is easy to explain in a live defense."
3. "Write a submission-ready README for a CPU-only 24-hour outage forecaster with a static HTML artifact."

## One discarded prompt

- Discarded prompt: "Build a full production-grade web app with real-time charts and an API backend."
- Why discarded: it was too large for the time budget and conflicted with the brief's low-bandwidth static UI requirement.

## Hardest decision

The hardest decision was choosing a model that was good enough for Tier 2 without becoming hard to defend live. I chose a simpler but well-structured baseline over a heavier model because the brief rewards a real, reproducible, explainable pipeline on CPU. That trade-off also made it easier to produce the low-bandwidth UI and the product artifact with time left for documentation.
