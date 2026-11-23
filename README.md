# Population information and reliable reporting of educational attainment gaps

Code and aggregate-data release **1.2.1** for the manuscript by **Runqing Liao and Zhining Wang**. Corresponding author: Zhining Wang, mathwangzhining@163.com. Both authors: School of Mathematics and Statistics, Hanshan Normal University, Chaozhou 521041, China. The manuscript is in preparation, not a peer-reviewed publication.

We ask when finite household samples support a directional or within-band statement about the age–sex standardized BA-or-higher attainment gap between foreign-born and native adults aged 25+. The targets are fixed public ACS household respondent frames, not error-free resident-population truths. Mature allocation was useful; learned risk did not improve it consistently. This is an application and conditions-of-use study, not a new algorithm or causal explanation.

## Reproduce without respondent records

Use Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements.txt
python reproduce.py --verify --checks --tables --workdir reproduced_tables
```

This checks release hashes, runs exact small-population estimator/design checks, verifies nine point/interval metrics for all **768 state–budget–method–endpoint rows**, and regenerates **four v8 descriptive CSVs** (counts exact; floating values within rtol=2e-12 and atol=2e-14 across platforms; all four byte-identical in the original Windows environment) from the **24 frozen simulation-output files**. Those files have 2,000 repetitions per state/budget and estimates for 16 methods. A row is a simulation repetition, not a respondent. It does **not** redraw the 2024 household samples, refit a model, or constitute another final evaluation.

To reconstruct a descriptive target directly from the authoritative Census release:

```sh
python verify_raw_context.py --state ct
# or use a previously downloaded official file:
python verify_raw_context.py --state ct --file /path/to/acs_2024_ct.zip
```

The program verifies the original source SHA256 and reproduces eligible household/adult counts and the raw and standardized gaps. It fails on a revised source rather than silently substituting it. This verifies outcome definitions, not original-survey replicate uncertainty or an entire training pipeline.

## Files and reproducibility boundaries

| Path | Content |
|---|---|
| `scripts/acs183`–`acs188` | Original data, risk-learning, design, estimator, evaluation and figure implementations |
| `scripts/acs49_*`, `acs26_models.py` | Earlier source-predictor dependencies |
| `scripts/acs201*`, `acs202*` | Post-score descriptive analysis and figure code |
| `data/tables/` | All 14 source tables, including unfavorable results and a header-only no-match table |
| `data/saved_draws/` | Frozen aggregate simulation estimates/variances/counts, not selected unit IDs |
| `metadata/frozen/` | Original numerical contracts, development selection, standard population and freeze hashes |
| `metadata/census_sources.json` | Source URLs, acquisition dates where documented, sizes and checksums; no HTTP session headers |
| `figures/` | Three main and two supplementary vector figures |

## Rebuild the used ACS pipeline from official raw files

**Reference platform: Windows, Python 3.12 and the pinned dependencies.** Strict refitting parity is validated locally and on an independent GitHub Windows runner: [raw reconstruction run 44992938047](https://github.com/liaorunqing/education-gap-reporting/actions/runs/44992938047). All 16 methods have zero point/variance differences and identical resources. The additional Linux audit agrees for 12 methods but not four learned-risk CPS variants; the largest per-draw point difference is about 0.147. This is not a CSV-format issue and the strict comparison is not relaxed. Use the manual raw workflow's platform selector to inspect either environment. See `docs/NUMERICAL_PORTABILITY.md`.

```sh
# About 704 MB of official compressed inputs; no pretrained models are used.
python scripts/acs222_fetch_sources.py --raw-dir data/raw
python scripts/acs216_raw_reconstruction.py --raw-dir data/raw --workdir reconstructed_CT
```

This rebuilds the used path for **CT 2024, 1,000 households, all 2,000 repetitions and 16 methods**: the 2017 national training sample, source HGB, historical cell control, eight-state 2022 standard population and household risk models, current background/design, then estimates and intervals. Original current education is opened only after the rebuilt prediction/design freeze. The saved reference is used only after rebuilding. In the original Windows environment all 29 compared fields, including points, variances, resources and counts, matched exactly (maximum difference 0).

This does not reconstruct unused source learner families, risk cross-validation diagnostics, every state/budget, or the entire historical research programme. Models and person/household arrays are created locally in the NEW work directory and are not distributed. Keep raw/private caches out of commits. Python/environment differences may affect refitted models; all reference discrepancies must be reported, not silently retuned.

## Real-panel feasibility pilot (separate from ACS confirmation)

```sh
python scripts/acs223_sipp_reproduce.py --fetch --raw-dir data/raw/sipp --workdir reconstructed_SIPP
```

This uses official **2024 and 2025 SIPP** files and the frozen plan under `metadata/sipp/`. It audits prior-wave roster retention and runs five mature controls at 300/1,000 baseline household clusters, with 2,000 repetitions. It uses prior weights and prior education/background; follow-up weights do not enter the designs. The target is the **retrospectively fully observed baseline cohort**, not current residents and not a frame known before fieldwork. It cannot establish field cost savings or a complete current roster. The 2024 December prior reference is frozen before 2025 education parsing.

The new source shows a useful historical predictor but incomplete follow-up and imperfect interval coverage. About 61% of eligible baseline adults/households have complete follow-up under the stated definitions; qualification reversals and background inconsistencies remain visible. At 1,000 clusters, prior-BA prediction greatly reduces conditional MSE while ordinary normal intervals cover only 89.8%; mature working allocation improves this to 94.0%. This is an observed limitation of these first-order intervals and this cohort, not proof of a new statistical principle. Reported-only sensitivity changes the target subset; it is not independent degree verification. See `data/sipp_audit/`, `data/sipp_pilot/`, and [conditions of use](docs/CONDITIONS_OF_USE.md).

The public self-contained entry points above are tested separately from the original legacy scripts. See [reproducibility boundaries](docs/REPRODUCIBILITY.md).

Original records are available from [Census ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html) and the [official file directory](https://www2.census.gov/programs-surveys/acs/data/pums/). No household or individual records, PUMS/SIPP person identifiers, trained pickle files, acquisition cookies, private logs or third-party article PDFs are distributed. Both raw workflows create private derived arrays locally; those caches are excluded from this release. See [release audit](docs/DATA_RELEASE.md).

## Practical interpretation

The large allocation gain assumes current linked background information and supplied demographic totals. A Census address frame does not establish those inputs. Panel documentation shows that prior information can be reused, but does not validate completeness, costs, attrition adjustment or this design. [Conditions of use](docs/CONDITIONS_OF_USE.md) separates what could be implemented from what remains unverified. The 2-percentage-point band is reporting resolution, not a validated policy threshold. No resident-group ranking, individual prediction service or resource-allocation rule is deployed.

## Licence, citation and funding

Code: **MIT**. Author-generated aggregate results, figures and documentation: **CC BY 4.0**, with third-party rights reserved (`LICENSE-DATA`). Underlying Census data retain their original status. This research product is not endorsed or certified by the Census Bureau. Cite `CITATION.cff` and the versioned [GitHub release](https://github.com/liaorunqing/education-gap-reporting/releases/tag/v1.2.1). A Zenodo metadata file is prepared; **no Zenodo deposit or DOI is claimed**.

This work was supported by the National Natural Science Foundation of China (No. 12271132) and the fund provided by the Department of Education of Guangdong Province, China (No. 2024ZDZX1027).

OpenAI Codex assisted code development, checks and documentation; this does not replace author review or imply confirmed author contribution roles. See the manuscript's final author declarations when available.

Timing qualification: the 2024 SIPP public file was released on 24 July 2025, after the February–June 2025 follow-up fieldwork. Earlier-reference final weights are not thereby proven available before selection. Historical responses can precede recontact internally, but this retrospective public-file pilot does not validate an as-of operational extract or preselection availability of final weights. See data/sipp_audit/availability_timing_audit.json.

To regenerate Supplementary Figure S3 from released aggregate CSVs only, run `python scripts/acs226_sipp_figure.py`. Output goes to the ignored `reproduced_figures` directory.


## Extension in 1.2.1: interval and membership diagnostics

`python reproduce_extension.py` checks all extension table denominators, collects 20 case summaries, reproduces membership-summary values, and checks finite-population bootstrap conditional moments. This is an audit from saved case summaries, not an independent rerun of their inner bootstrap samples. The original `reproduce.py` still replays the ACS simulation outputs.

Extension inputs/outputs: `data/followup/`, `data/interval_reference/`, `metadata/followup/`. All unfavorable methods, two scopes, two budgets, confidence levels and report resolutions are retained. The original membership plan has 40 rows; 30 additional calibrated controls were fixed after inspecting those results and are explicitly exploratory. The original statements and source archive versions remain frozen.

Full numerical extension after the documented raw SIPP workflow:

```sh
python scripts/acs223_sipp_reproduce.py --fetch --raw-dir data/raw/sipp --workdir reconstructed_SIPP
python scripts/acs242_followup_plan.py --pilot-dir reconstructed_SIPP --baseline-dir data/interval_reference --out-dir followup_replay
python scripts/acs243_bootstrap_intervals.py --pilot-dir reconstructed_SIPP --out-dir followup_replay
python scripts/acs244_membership_correction.py --pilot-dir reconstructed_SIPP --out-dir followup_replay
python scripts/acs244_membership_correction.py --pilot-dir reconstructed_SIPP --out-dir followup_replay --strong-sensitivity
python scripts/acs247_missing_change_sensitivity.py --plan --out-dir followup_replay
python scripts/acs247_missing_change_sensitivity.py --pilot-dir reconstructed_SIPP --out-dir followup_replay
```

Choose NEW work directories and keep respondent caches private. Main inner bootstrap size is 999; the first 100 draws per case also use 3,999. Parameters and cases are not searched. Bootstrap is approximate; empirical Bernstein-Serfling bounds concern the fixed retained frame only. No respondent records, trained models or contact information are redistributed. A fresh technical replay plan is not the historical plan's creation date. See `docs/SIPP_QUALITY_AND_INTERPRETATION.md`. No independent population confirmation, identified missing future degrees, field-cost savings or new method is claimed. This release is software and aggregate evidence; it does not publish the draft manuscript or imply final all-author approval.

Version 1.2.1 is a metadata-only correction: all 12 historical protection entries retain distinct relative paths. Version 1.2.0 shortened them to basenames and collapsed one duplicate. No numerical dataset, method, seed, result or source-file checksum changed; the issued 1.2.0 tag remains untouched.
