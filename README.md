# Population information and reliable reporting of educational attainment gaps

Code and aggregate-data release **1.0.0** for the manuscript by **Runqing Liao and Zhining Wang**. Corresponding author: Zhining Wang, mathwangzhining@163.com. Both authors: School of Mathematics and Statistics, Hanshan Normal University, Chaozhou 521041, China. The manuscript is in preparation, not a peer-reviewed publication.

We ask when finite household samples support a directional or within-band statement about the age–sex standardized BA-or-higher attainment gap between foreign-born and native adults aged 25+. The targets are fixed public ACS household respondent frames, not error-free resident-population truths. Mature allocation was useful; learned risk did not improve it consistently. This is an application and conditions-of-use study, not a new algorithm or causal explanation.

## Reproduce without respondent records

Use Python 3.12 in a virtual environment:

```sh
python -m pip install -r requirements.txt
python reproduce.py --verify --checks --tables --workdir reproduced_tables
```

This checks release hashes, runs exact small-population estimator/design checks, verifies nine point/interval metrics for all **768 state–budget–method–endpoint rows**, and regenerates **four v8 descriptive CSVs byte-for-byte** from the **24 frozen simulation-output files**. Those files have 2,000 repetitions per state/budget and estimates for 16 methods. A row is a simulation repetition, not a respondent. It does **not** redraw the 2024 household samples, refit a model, or constitute another final evaluation.

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

`scripts/` preserves the original relative directory conventions. The full raw-to-model pipeline also needs the 2017 training sample, source model generation, history, and derived household/person arrays. These are **not** bundled here. The script-level source is provided for inspection, but only the two entry points above have a self-contained tested public workflow. The original full local replay package has more artifacts; it is not silently equated with this disclosure-minimised public release. See [reproducibility boundaries](docs/REPRODUCIBILITY.md).

Original records are available from [Census ACS PUMS](https://www.census.gov/programs-surveys/acs/microdata.html) and the [official file directory](https://www2.census.gov/programs-surveys/acs/data/pums/). No household or individual records, PUMS identifiers, trained pickle files, acquisition cookies, private logs or third-party article PDFs are distributed. See [release audit](docs/DATA_RELEASE.md).

## Practical interpretation

The large allocation gain assumes current linked background information and supplied demographic totals. A Census address frame does not establish those inputs. Panel documentation shows that prior information can be reused, but does not validate completeness, costs, attrition adjustment or this design. [Conditions of use](docs/CONDITIONS_OF_USE.md) separates what could be implemented from what remains unverified. The 2-percentage-point band is reporting resolution, not a validated policy threshold. No resident-group ranking, individual prediction service or resource-allocation rule is deployed.

## Licence, citation and funding

Code: **MIT**. Author-generated aggregate results, figures and documentation: **CC BY 4.0**, with third-party rights reserved (`LICENSE-DATA`). Underlying Census data retain their original status. This research product is not endorsed or certified by the Census Bureau. Cite `CITATION.cff` and the versioned [GitHub release](https://github.com/liaorunqing/education-gap-reporting/releases/tag/v1.0.0). A Zenodo metadata file is prepared; **no Zenodo deposit or DOI is claimed**.

This work was supported by the National Natural Science Foundation of China (No. 12271132) and the fund provided by the Department of Education of Guangdong Province, China (No. 2024ZDZX1027).

OpenAI Codex assisted code development, checks and documentation; this does not replace author review or imply confirmed author contribution roles. See the manuscript's final author declarations when available.
