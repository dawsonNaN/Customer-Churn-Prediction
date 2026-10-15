# Customer Churn Prediction and Retention Analysis

A complete customer analytics project covering exploratory analysis, cleaning, feature engineering, churn modeling, model interpretation, risk scoring, and customer segmentation.

The business context is inspired by cloud subscription services such as Microsoft Azure. The analysis uses Telco data as a proxy, together with a separate customer support dataset. These are not actual Azure customer records, and the project does not include an Azure deployment.

## Project highlights

| Area | Work included |
| --- | --- |
| Data exploration | Two English EDA notebooks and a Power BI dashboard. |
| Data preparation | A consolidated module for cleaning and engineering both datasets. |
| Baseline models | Logistic Regression and Decision Tree under five imbalance settings. |
| Advanced models | XGBoost, LightGBM, and CatBoost, Optuna tuning, threshold selection, and probability averaging. |
| Interpretation | Global and local SHAP, local LIME, and customer risk scorecards. |
| Segmentation | RFM-inspired value scores, K-Means/DBSCAN comparison, and a nine-cell value/risk matrix. |
| Reporting | Nine English Markdown reports, with matching PDFs. |

## Main results from the original project

The main Telco dataset contains 7,043 customers, including 1,869 churn cases, or about 26.5%. The final supplied matrix has 44 predictors plus the churn target. The separate support dataset has 8,469 tickets without a churn label or a shared Telco customer ID.

| Result | Original reported value | Context |
| --- | --- | --- |
| Balanced Logistic Regression validation ROC-AUC | 0.7882 | Chinese Week 5 evaluation report; the separate Markdown source gives 0.7907. |
| CatBoost validation ROC-AUC | 0.7999 | Original Week 6-7 reports; saved model supplied. |
| CatBoost classification threshold | Approximately 0.64 | Stored in the supplied model bundle. |
| High-risk customer count | 1,845 | Supplied full-sample Week 7 scorecard. |
| Selected K-Means setting | k=4; silhouette 0.4569 | Original Week 8 results. |
| Core high-value K-Means cluster | 1,732 customers; 61.47% of TotalCharges | Observed accumulated spending, not profit. |
| Top value/risk cells reaching 60%+ | Three groups; 67.59% of TotalCharges | Original Week 8 contribution analysis. |

Month-to-month contracts, shorter tenure, lack of technical support, and electronic check payment are associated with higher observed churn. The interpretation and segmentation stages connect those patterns to possible retention priorities. No causal effect of an intervention or realized financial return is claimed.

![Observed churn by contract](assets/figures/churn_by_contract.png)

## Final repository layout

| Path | Contents |
| --- | --- |
| [run_pipeline.py](run_pipeline.py) | One command-line entry point for every stage. |
| [src](src) | Shared paths/metrics plus preparation, baseline, advanced, interpretation, and segmentation modules. |
| [notebooks](notebooks) | Three English notebooks for Telco EDA, support EDA, and interpretation. |
| [data/processed](data/processed) | Supplied Telco working snapshots and a reduced public support table. |
| [models](models) | Original fitted CatBoost bundle and a readable metadata sidecar. |
| [reports](reports) | Nine Markdown reports; matching PDFs in reports/pdf. |
| [results/reported](results/reported) | Original report tables, supplied Week 7 scorecard, and all six Week 8 CSV outputs. |
| [results/reproduced](results/reproduced) | Separate verification outputs from the final consolidated code. |
| [assets/figures](assets/figures) | English figures for the reports and project overview. |
| [dashboards](dashboards) | Original Power BI native file and connection notes. |
| [docs](docs) | Feature dictionaries, metrics, file map, run notes, and refactoring notes. |

Generated working files go under outputs. That folder is ignored by Git. Supplied data and model snapshots are not overwritten by a normal run.

## What was consolidated

The four original Telco/support cleaning and feature scripts are combined in src/prepare_data.py. Shared dataset paths, lifecycle splitting, evaluation metrics, and threshold selection live in src/project_paths.py. Modeling, interpretation, and segmentation remain separate modules because they serve different stages. The final project uses one runner instead of several download-dependent weekly entry scripts.

See [refactoring notes](docs/refactoring_notes.md) for the exact mapping and the small reproducibility changes.

## Install and run

Use Python 3.11 or later. From the repository root, create an environment:

```bash
python -m venv .venv
```

Activate it with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on macOS/Linux, then install the core analysis packages:

```bash
python -m pip install -r requirements.txt
```

These stages use the included local data and supplied risk scorecard. They do not require a new dataset download:

```bash
python run_pipeline.py prepare
python run_pipeline.py baseline
python run_pipeline.py segment
```

To use the supplied model for SHAP/LIME interpretation, or to retrain the advanced models, install the optional packages:

```bash
python -m pip install -r requirements-advanced.txt
python run_pipeline.py interpret
python run_pipeline.py advanced --trials 10
```

To run every stage as one connected workflow:

```bash
python run_pipeline.py all --trials 10
```

The full workflow uses its newly prepared matrix for modeling, saves newly trained advanced models, explains the newly trained CatBoost specifically, and passes its new scorecard into segmentation. A different model or ensemble can lead the new AUC ranking; the interpretation stage still names CatBoost explicitly rather than treating it as the automatic winner.

Standalone baseline/advanced stages use the supplied matrix by default. Standalone interpret uses the supplied CatBoost model. Standalone segment uses the supplied Week 7 scorecard. Options include --data, --model, --scorecard, --seed, --trials, --output-dir, and --no-lime; see `python run_pipeline.py --help`.

For the notebooks, install `requirements-notebooks.txt` and open JupyterLab from the repository root. The third notebook also needs the advanced requirements. It reuses the interpretation module rather than keeping a second copy of the same analysis logic.

## Read the reports

1. [Business context, data quality, and roadmap](reports/01_business_context_and_data_quality.md)
2. [Exploratory findings](reports/02_exploratory_findings.md)
3. [Feature engineering design](reports/03_feature_engineering_design.md)
4. [Preliminary feature importance](reports/04_preliminary_feature_importance.md)
5. [Baseline models and evaluation](reports/05_baseline_models_and_evaluation.md)
6. [Separate baseline source summary](reports/06_baseline_run_summary.md)
7. [Advanced model evaluation](reports/07_advanced_model_evaluation.md)
8. [Model interpretation and risk scorecard](reports/08_model_interpretation_and_risk_scorecard.md)
9. [Customer segmentation and value](reports/09_customer_segmentation_and_value.md)

![Value contribution by risk and value level](assets/figures/week8_value_risk_matrix.png)

## Interpretation limits

Tenure sorting creates a lifecycle split, not a calendar-time forecast. The original matrix was preprocessed before splitting, and the exploratory feature rankings used all labels. Full-sample risk tables include training customers. These choices limit claims about performance on future customers.

The customer value method is RFM-inspired: tenure is not transaction recency, service_count is not purchase frequency, and TotalCharges is spending rather than profit. Support tickets cannot be joined to the Telco customers. Their purchase-to-first-response interval is not a support response-time SLA.

The supplied PKL and historical scorecard do not produce the same risk scores. Standalone segmentation deliberately uses the supplied scorecard to reproduce Week 8; newly computed model scores are stored separately. The supplied reports retain their original scores. Seeded reruns can give different advanced tuning and LIME results, and those outputs are recorded separately. The advanced targets of AUC >= 0.88 and F1 >= 0.80 were not reached in the original reported results.

The Power BI file remains in its original native format, with a few Chinese automatic aggregation labels. It was not opened or refreshed in Power BI Desktop. See [dashboard notes](dashboards/README.md), [data notes](docs/data_notes.md), and [reproduction notes](docs/reproduction_notes.md).
