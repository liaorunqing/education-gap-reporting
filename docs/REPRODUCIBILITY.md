# What can be reproduced here

1. Release file integrity using `reproduce.py --verify`.
2. Exact finite-population enumeration checks for sampling/estimators, including two-wave conditional unbiasedness/variance and a perturbation test that changing unobserved labels cannot change selection, using `--checks`.
3. Nine numerical point/interval metrics for all 768 saved state–budget–method–endpoint rows, also using `--checks`.
4. Four post-score correct-yield CSVs (counts exact; floats within rtol=2e-12, atol=2e-14 across platforms; byte-identical in the original Windows environment), using `--tables` with a new work directory.
5. Counts and raw/standardized descriptive gaps directly from each official 2024 state ZIP, using `verify_raw_context.py`. Acquisition SHA256 values prevent unannounced vintage substitution.

This is not a full original training-pipeline reconstruction or household-level resampling replay. The first 8 CT household draws were exactly replayed locally in the prior private review package across all 16 methods, but the required unit-level artifacts are excluded from this public release. That historical test is not silently claimed as a public bundled workflow.

The scientific implementation and its import dependencies are available under `scripts/`. Old paths and freeze guards are preserved. Rebuilding models requires reconstructing the 2017 training frames and historical/current records from Census, including the initial sampling plans and predictor artifacts. This has not been newly verified as a one-command workflow on another machine. Do not run legacy modes `freeze` or `final` in place and describe the outcome as another unseen confirmation.

Sources/tables/figures are fixed to the manuscript's original calculations. The v8 reporting-yield re-expression was post-score. Author/declaration and open-release changes do not reopen model selection or the final labels. Synthetic exact checks validate implementation, not substantive generalization. No claim of independent external validation, national random-state inference, field cost savings or degree-label truth follows from this release.
