# Week 5 Baseline Run Summary

## Data used

This summary follows the separately supplied English baseline Markdown report. The input is the Week 4 model-ready Telco matrix, with 7,043 rows, 44 predictors, and the Churn_binary target. It includes cleaning, feature engineering, one-hot encoding, standardization, and correlation filtering.

## Data partitioning

Customers are sorted by tenure in descending order, then divided into 60% training, 20% validation, and 20% test data.

| Partition | Rows | Columns, including target |
| --- | ---: | ---: |
| Training | 4,225 | 45 |
| Validation | 1,409 | 45 |
| Test | 1,409 | 45 |

This is a tenure-based lifecycle split. There are no actual observation dates, so the split does not establish a historical-to-future forecasting sequence.

## Models and imbalance handling

Logistic Regression and Decision Tree models are compared under five settings: no resampling, balanced class weights, random oversampling, SMOTE, and ADASYN. The evaluation uses accuracy, precision, recall, F1-score, and ROC-AUC, with validation AUC used for selection.

## Best setting in this source version

The supplied Markdown summary identifies Logistic Regression with balanced class weights as the best validation setting.

| Metric | Validation | Test |
| --- | ---: | ---: |
| ROC-AUC | 0.7907 | 0.7479 |
| Precision | 0.4531 | 0.6331 |
| Recall | 0.9354 | 0.9007 |
| F1-score | 0.6105 | 0.7436 |

The validation AUC exceeds the project's baseline target of 0.75. The test AUC in this source version is slightly below that target, so the target should not be described as having been achieved on every partition.

## Interpretation

The engineered features contain useful churn information in this prototype. Balanced weights place more emphasis on churn cases and produce high recall, with a tradeoff in false positives. Future work could compare more flexible models and choose a classification threshold based on retention costs and capacity.

These values differ slightly from the Chinese Week 5 evaluation PDF and the baseline values quoted in Week 6. Both source versions are preserved, and newly reproduced results are stored separately. Full-dataset preprocessing and the absence of genuine observation dates limit the strength of the performance claim.
