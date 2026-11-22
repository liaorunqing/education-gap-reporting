# Population information and reliable reporting of educational attainment gaps

Code and aggregate-data release **1.1.0** for the manuscript by **Runqing Liao and Zhining Wang**. Corresponding author: Zhining Wang, mathwangzhining@163.com. Both authors: School of Mathematics and Statistics, Hanshan Normal University, Chaozhou 521041, China. The manuscript is in preparation, not a peer-reviewed publication.

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

Code: **MIT**. Author-generated aggregate results, figures and documentation: **CC BY 4.0**, with third-party rights reserved (`LICENSE-DATA`). Underlying Census data retain their original status. This research product is not endorsed or certified by the Census Bureau. Cite `CITATION.cff` and the versioned [GitHub release](https://github.com/liaorunqing/education-gap-reporting/releases/tag/v1.1.0). A Zenodo metadata file is prepared; **no Zenodo deposit or DOI is claimed**.

This work was supported by the National Natural Science Foundation of China (No. 12271132) and the fund provided by the Department of Education of Guangdong Province, China (No. 2024ZDZX1027).

OpenAI Codex assisted code development, checks and documentation; this does not replace author review or imply confirmed author contribution roles. See the manuscript's final author declarations when available.

Timing qualification: the 2024 SIPP public file was released on 24 July 2025, after the February–June 2025 follow-up fieldwork. Earlier-reference final weights are not thereby proven available before selection. Historical responses can precede recontact internally, but this retrospective public-file pilot does not validate an as-of operational extract or preselection availability of final weights. See data/sipp_audit/availability_timing_audit.json.

To regenerate Supplementary Figure S3 from released aggregate CSVs only, run `python scripts/acs226_sipp_figure.py`. Output goes to the ignored `reproduced_figures` directory.
