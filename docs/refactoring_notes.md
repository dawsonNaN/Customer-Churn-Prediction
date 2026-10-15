# Refactoring Notes

## Original code to final modules

| Original work | Final location |
| --- | --- |
| Telco cleaning and Telco feature engineering scripts | src/prepare_data.py: clean_telco and engineer_telco. |
| Support cleaning and support feature engineering scripts | src/prepare_data.py: clean_support and engineer_support. |
| Week 5 baseline training | src/train_baselines.py. |
| Week 6 advanced tuning | src/train_advanced.py. |
| Week 7 interpretation notebook computations | src/interpret_model.py, reused by notebooks/03_model_interpretation.ipynb. |
| Week 8 ZIP analysis script | src/segment_customers.py. |
| Repeated paths, split, metrics, and threshold search | src/project_paths.py. |
| Separate weekly entry points | run_pipeline.py. |

There are five analysis modules, one shared helper module, and one runner. Stages have functions and do not start training when imported. The first four original scripts are merged rather than retained as duplicate entry points.

## Preserved analytical choices

Feature formulas, high-charge medians, standardization rules, the correlation cutoff, feature-ranking approaches, baseline settings, advanced search ranges, value-score weights, clustering settings, and risk-score boundaries follow the supplied work. The full-dataset preprocessing and tenure-based partition are retained for reproduction and explicitly documented as limitations.

## Small implementation changes

- The mutual-information estimator now has a seed, and selected features follow matrix column order instead of arbitrary set order. This changes possible selected-feature rankings, not the full 44-predictor matrix.
- LIME uses highest-weight feature selection to avoid a singular unregularized forward-selection fit on constant encoded features in the tested software environment. This is a documented implementation change; original LIME weights remain historical values.
- Optuna and LIME are seeded in the final edition. The original scripts did not seed those random components. New results are kept separate from historical reports.
- Advanced model ranking uses full-precision validation AUC instead of selecting after rounding it to four decimals.
- Boolean dummy columns retain their original baseline dtype so SMOTE/ADASYN behavior matches the supplied script. The advanced stage retains its original explicit conversion of booleans to integers.
- Common classification metrics and the 0.10-0.90 F1 threshold grid are shared. Baseline results use short confusion-matrix names tn/fp/fn/tp; earlier exported files used longer names.
- Training uses bounded thread counts. This changes resource use and can contribute to small numerical differences across platforms.
- A newly tuned CatBoost can select a threshold at or below 0.35. For that case only, the low-risk boundary becomes half the decision threshold so the risk bands remain non-overlapping. The supplied model retains the original 0.35/0.64 boundaries.
- The advanced stage saves an explicit CatBoost explanation bundle even if another candidate leads the new AUC ranking. The connected workflow interprets that named model.
- Segmentation checks one-to-one row coverage and actual churn alignment before joining the risk scorecard. Customer indices still depend on original row order.
- The original Week 8 marker rendering is adjusted so every group remains visible in black and white; clustering calculations are unchanged.
- Names, emails, ticket descriptions, and resolution free text remain excluded from public support outputs.

## Notebook changes

The first two EDA notebooks retain their English analysis and local loaders. The third removes Colab upload prompts and installation cells, and calls the shared interpretation module. Its tables and figures are displayed in the notebook. The original notebook's local SHAP table is no longer misleadingly called LIME.
