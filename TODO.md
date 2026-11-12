# PeakGuard AI Implementation Plan
Goal: standalone Thai Google Colab notebook, one-hour-ahead PJME peak classification.
Scope: R1-R7 notebook, docs, provenance, poster and local checks. Parent notebook was intentionally deleted by the user; do not restore or inspect it. No push, deployment, web app, secrets or fake results.
Architecture: embedded notebook functions; hourly grid; chronological split; train-only P90; validation-only selection; frozen final test.
Stack: Python 3.12+, pandas, numpy, matplotlib, seaborn, scikit-learn, joblib. Jupyter packages for local verification.

## Tasks and acceptance
- [x] Inspect workspace and runtime. Public Kaggle metadata/download obtained; license field says CC0: Public Domain.
- [x] Implement causal data contract and focused checks: initial 19 tests passed; current local suite passes 33 tests including portability checks. Initial missing implementation and later timedelta warning reproduced before fixes.
- [x] Build R1-R7 notebook with three loader modes plus local, train EDA, two baselines, Logistic Regression, Random Forest, frozen selection and errors/metrics. Interactive account-dependent modes remain untested.
- [x] Execute real data Notebook on local Jupyter CPU: 22 code cells, no cell errors, 14 PNG outputs; not a live Colab run.
- [x] Verify deliverables: 61 project-only artifact checks, recomputed metrics, reload predictions and 300-DPI metadata.
- [x] Bounded independent correctness review: scoped PASS after Kaggle stale-download finding reproduced, fixed and re-reviewed; see validation/INDEPENDENT_REVIEW.md for coverage limits.

## Decisions and risks
- Preserve naive source labels; no UTC/DST assumption. Identical duplicate readings collapse; conflicting duplicate hours become missing.
- Reindex hourly, mark nonfinite/nonpositive readings missing, no interpolation/backfill. Retain extremes; train-only statistical outlier audit.
- Prediction at t assumes completed observation t is available. All inputs must be at or before t.
- Split raw grid 70/15/15 before feature filtering. Purge samples with t and t+1 in different partitions. Missing values do not bridge gaps.
- Train-only P90/scaler/model fit. Validation chooses F2, then Recall, then Precision with deterministic ties. Do not refit before test.
- Final test compares frozen models; it never changes the validation-selected winner. Average Precision is the chosen PR-AUC summary.
- Success is reproducible evidence and correct evaluation, not guaranteed baseline improvement.
- Recovery: preserve all user work and the user-confirmed parent notebook deletion. A Git repository now exists at the workspace root; do not commit or reset it.

## Review focus
Exact one-hour target through gaps; conflicting duplicate hours; future-perturbation invariance; one-class failure; honest poster/AI log.

## Evidence and limitations (2026-11-12)
No course PDF or existing tracker found. Python 3.12.15 via uv. Dependencies installed in workspace environment; exact versions in validation/environment-freeze.txt.
Native update_plan is not available; this file is the authoritative tracker. Independent read-only review is authorized; no worker edits or subdelegation.

## Follow-up local validation (2026-11-12)
- [x] Preserve user-edited R1-R6 headings; align the generator and verify regeneration only in a temporary project. Original notebook bytes unchanged.
- [x] Exercise upload, Drive and Kaggle loader success/failure paths with local fixtures and mocked external boundaries. No account access, network or credential use.
- [x] Run all 33 local tests and 61 artifact checks; notebook SHA-256 unchanged. Current evidence: validation/followup_report.json and validation/followup-tests.txt. Live-Colab acceptance remains open.

## Final acceptance gaps
- [x] Independent core-flow review passed within supplied scope; full rendered/prose review not claimed.
- [?] Course PDF comparison requires the missing PDF.
- [?] Team must supply actual R7 decision history including a real rejected/amended AI recommendation.
- [ ] Account-dependent Colab upload/Drive/Kaggle smoke checks require execution in the user account. Local Jupyter equivalent was executed.

## Execution ledger
- Public source file: 145366 raw rows; 4 conflicting duplicate hours; 30 missing grid hours; no invalid demand rows.
- Train P90: 40851 MW; usable Train/Validation/Test: 97525/20958/20958.
- Validation selected Random Forest at cutoff 0.375; final Test F2 0.975446, Recall 0.995787, FP 206, FN 8.
- Visual correction moved legend outside model-comparison axes; no tuning from test outcomes.
- Explicit NumPy time units remove observed Pandas/NumPy timedelta deprecation; regression evidence retained.
- Main implementation; independent reviewer completed direct-context bounded review after initial tool-access blocker. Graphify external graph construction was not run: no need to transmit project content or add graph artifacts for this notebook.
- No commits, pushes, deployment, access changes or billable data providers. Overall status PARTIAL until acceptance gaps are resolved.
- Follow-up: user confirmed removing Phase labels from R1-R6. Generator now matches those headings. R7 can record this actual amendment only after the user supplies its reason and confirms team attribution.
