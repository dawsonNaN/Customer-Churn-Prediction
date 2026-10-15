# Reproduction and Editorial Notes

## What was checked

The consolidated preparation, baseline, advanced, interpretation, and segmentation stages ran using local files. The full Telco matrix matches the supplied 7,043 by 45 matrix exactly in parsed values and column order. All 51 support feature columns match the earlier full feature script, allowing only numeric serialization tolerance.

The baseline reproduces the original code metrics. A fresh seeded advanced-model run completed ten Optuna trials for each of XGBoost, LightGBM, and CatBoost. SHAP, three local LIME explanations, and a new full customer scorecard were computed from the supplied model. All six Week 8 CSVs match the supplied archive outputs within floating-point serialization tolerance when using the supplied Week 7 scorecard.

The first two English EDA notebooks were checked in the earlier edition. The new interpretation notebook executes the shared interpretation module and has its tables and plots embedded. Notebook code is checked sequentially in Python; the interactive Jupyter interface is not tested. The native Power BI file was not opened or refreshed.

## Baseline parity

| Partition | ROC-AUC | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: |
| Validation | 0.788194 | 0.452642 | 0.941909 | 0.611448 |
| Test | 0.750117 | 0.632935 | 0.910053 | 0.746609 |

These results agree with the Chinese Week 5 PDF to four decimals. The separate supplied English baseline summary has validation AUC 0.7907 and test AUC 0.7479; that source version remains in Report 06.

## Source model and scorecard are different versions

The supplied model bundle loads successfully, has 212 CatBoost trees and 44 named features, and stores a threshold near 0.64. It does not reproduce the separately supplied historical Week 7 scorecard.

Across 7,043 customers, the maximum absolute probability difference is 0.223263, the mean absolute difference is 0.080878, and 1034 risk labels differ. The exact reason is not established from the supplied files.

| Source | Low Risk | Medium Risk | High Risk |
| --- | ---: | ---: | ---: |
| Supplied scorecard | 2190 | 3008 | 1845 |
| Recomputed supplied model | 2968 | 2032 | 2043 |

The original risk CSV and model binary are preserved unchanged. Standalone segmentation deliberately uses the original scorecard so that the Week 8 results remain reproducible. Recomputed model scores are stored under supplied_model_interpretation and do not overwrite historical scores. The report SHAP table remains the original reported table, while its newly computed figure is explicitly labeled as a different source version.

## New advanced tuning run

The new seed-42, ten-trial run selects CatBoost with validation AUC 0.797444. These are new results, not a reconstruction of the original unseeded Optuna history. Full validation/test metrics, trial records, parameter summaries, and segment metrics are stored under advanced_tuning.

The connected workflow is separately checked with one trial per model and LIME skipped. This exercises stage handoffs rather than rerunning the full tuning budget. Its result is recorded in connected_workflow_check.json. The individual interpretation stage and notebook include the full three-customer LIME computation.

## Refactoring and compatibility changes

Four preparation scripts were combined into one module; repeated paths, splitting, metrics, and threshold search were moved into shared helpers. The final runner connects newly prepared data, trained models, a named CatBoost interpretation bundle, its scorecard, and segmentation.

Mutual information, Optuna, and LIME now use fixed seeds. Selected features follow matrix order. Advanced selection uses full-precision AUC. LIME uses highest-weight feature selection because its default unregularized forward-selection step produced a singular fit on constant encoded columns in the tested environment. If a newly trained CatBoost threshold is at or below 0.35, the low-risk boundary becomes half that threshold; the supplied 0.35/0.64 boundaries remain unchanged.

## Editorial clarifications

- The data is not real Azure customer data, and no deployment is supplied.
- The first original report repeats TotalCharges values in the MonthlyCharges median and maximum cells. The corrected values are 70.35 and 118.75.
- Automatic bank-transfer churn rounds to 16.71% from the supplied data; the original overview used 16.70%.
- Tenure is not a date. Sorting it creates lifecycle groups and does not establish future forecasting validity.
- Full-dataset scaling, medians, correlation filtering, and exploratory feature ranking precede evaluation splitting. Reproduction does not repair this evaluation limitation.
- The feature narrative mentions an unimplemented is_mid_tenure_customer. The design document repeats the weekday field; the implemented weekend flag is documented instead.
- Pairwise correlation filtering does not establish the absence of every form of multicollinearity. Feature importance and SHAP are not causal effects.
- A support purchase-to-response interval is not a ticket response-time SLA. The original Valid label includes missing timestamps, and some boolean flags map unknown values to zero.
- Week 6 originally omitted some test AUCs; those cells remain unreported in the translated historical table. The Week 7 report adds a reported blending validation AUC of 0.7989 with its source noted.
- The original Week 7 lime_like_explanations variable actually holds local SHAP values. Actual LIME uses a separate explainer. Top risk-driver fields use absolute contributions and can include features that reduce a prediction.
- Risk-score summaries cover the full customer sample, including training rows; they are not independent validation results or calibration evidence.
- Week 8 uses a nine-cell matrix, despite four-quadrant wording in the task. Its RFM features are proxies, and charge contributions are not observed profit.
- The SaaS benchmark workbook is not supplied, so its quoted figures remain unverified context.

## Environment and recordkeeping

Python 3.12.14. Package versions, exact input hashes, score-version comparison, and verification details are recorded in run_metadata.json and the supplied model metadata. Core dependencies are pinned; the CPU XGBoost distribution used for verification exposes the same xgboost import used by the optional requirements.

The source-file mapping covers all 27 uploads. The Week 8 archive contributed one analysis script, six CSV files, and three PNG figures; macOS resource files and its local file shortcut were omitted.
