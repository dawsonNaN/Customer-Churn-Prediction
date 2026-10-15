# Week 6 Advanced Model Evaluation

## Purpose and available evidence

This report summarizes advanced model training, hyperparameter tuning, probability averaging, threshold selection, and customer segment analysis described in the original Week 6 report.

The reported input is the Week 4 model-ready Telco matrix. The original report states that XGBoost, LightGBM, and CatBoost were tuned using Optuna. The Week 6 training script and saved CatBoost model are now included. The original Optuna trial history and original per-customer prediction files were not supplied. The numbers below remain the original reported results; newly computed verification and tuning outputs are stored separately.

## Relationship to the baseline

Week 5 compared Logistic Regression and Decision Tree models under different imbalance strategies. The Week 6 report uses a balanced Logistic Regression validation AUC of 0.7882 as its baseline.

| Baseline model | Strategy | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | Balanced weights | 0.5905 | 0.4526 | 0.9419 | 0.6114 | 0.7882 |
| Decision Tree | None | 0.7246 | 0.6343 | 0.4606 | 0.5337 | 0.7860 |
| Logistic Regression | Random oversampling | 0.5862 | 0.4500 | 0.9440 | 0.6095 | 0.7846 |
| Decision Tree | Random oversampling | 0.6217 | 0.4723 | 0.9004 | 0.6196 | 0.7840 |
| Decision Tree | Balanced weights | 0.6828 | 0.5242 | 0.7863 | 0.6290 | 0.7564 |

Boosting models were explored because they can represent nonlinear patterns and feature interactions. The aim was to see whether they improved discrimination and churn identification over these baselines.

## Reported data split

As in Week 5, customers were sorted by tenure and divided into 60%, 20%, and 20% partitions.

| Partition | Rows | Columns including target | Reported churn rate |
| --- | ---: | ---: | ---: |
| Training | 4,225 | 45 | 0.1501 |
| Validation | 1,409 | 45 | 0.3407 |
| Test | 1,409 | 45 | 0.5358 |

These rates are copied from the Week 6 report. Sorting the supplied matrix in the current environment gives slightly different counts, documented in the reproduction notes. Records tied on tenure can be ordered differently, but the exact reason for the source differences is unverified.

Tenure is a lifecycle variable, not a calendar timestamp. The increasing churn prevalence reflects a difference between tenure groups created by the split; it is not evidence of an observed change over calendar time.

## Class imbalance

The advanced models reportedly used class weights or scale_pos_weight rather than synthetic oversampling. The report gives scale_pos_weight = 5.664, representing the training non-churn to churn ratio in that source run. This increases the relative training weight of churn observations without changing the number of rows.

## Optuna tuning

The stated optimization objective was validation ROC-AUC. Search areas included tree counts, depth, learning rate, sampling, leaf counts, and regularization. The report provides these parameter highlights. The supplied training script contains the search ranges; the saved CatBoost bundle contains the fitted model:

| Model | Best validation AUC | Reported parameter highlights |
| --- | ---: | --- |
| XGBoost | 0.7921 | n_estimators=169; max_depth=5; learning_rate=0.0118; min_child_weight=5 |
| LightGBM | 0.7951 | n_estimators=182; max_depth=2; learning_rate=0.0107; num_leaves=10 |
| CatBoost | 0.7999 | iterations=212; depth=4; learning_rate=0.0187; l2_leaf_reg=1.3488 |

CatBoost has the highest reported validation AUC, although the differences between the three models are small. Repeated tuning on a single validation set can also make that score optimistic.

## Threshold selection and reported performance

The models output churn probabilities, then apply a classification threshold. Instead of always using 0.50, the reported workflow chose thresholds using validation F1.

| Partition | Model | Threshold | Accuracy | Precision | Recall | F1 | AUC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Validation | XGBoost | 0.62 | 0.7367 | 0.5945 | 0.7146 | 0.6490 | 0.7921 |
| Test | XGBoost | 0.62 | 0.6608 | 0.7514 | 0.5483 | 0.6340 | Not reported |
| Validation | LightGBM | 0.55 | 0.7161 | 0.5571 | 0.8125 | 0.6610 | 0.7951 |
| Test | LightGBM | 0.55 | 0.6962 | 0.7183 | 0.7126 | 0.7154 | Not reported |
| Validation | CatBoost | 0.64 | 0.7410 | 0.5976 | 0.7333 | 0.6586 | 0.7999 |
| Test | CatBoost | 0.64 | 0.6508 | 0.7476 | 0.5258 | 0.6174 | Not reported |
| Validation | Probability average | 0.53 | 0.7104 | 0.5504 | 0.8187 | 0.6583 | Not reported |
| Test | Probability average | 0.53 | 0.6842 | 0.7286 | 0.6543 | 0.6895 | Not reported |

The source narrative rounds the probability-average validation recall to 0.8188 in places; the table gives 0.8187. The table value is used consistently here.

| Partition | Model | TN | FP | FN | TP |
| --- | --- | ---: | ---: | ---: | ---: |
| Validation | XGBoost | 695 | 234 | 137 | 343 |
| Test | XGBoost | 517 | 137 | 341 | 414 |
| Validation | LightGBM | 619 | 310 | 90 | 390 |
| Test | LightGBM | 443 | 211 | 217 | 538 |
| Validation | CatBoost | 692 | 237 | 128 | 352 |
| Test | CatBoost | 520 | 134 | 358 | 397 |
| Validation | Probability average | 608 | 321 | 87 | 393 |
| Test | Probability average | 470 | 184 | 261 | 494 |

CatBoost was selected using validation AUC. LightGBM has the highest reported test F1 among these advanced approaches, at about 0.7154. This does not change the validation-based selection rule. Missing test AUC values are left unreported rather than estimated from the confusion matrices.

## Probability averaging

The reported ensemble averages the predicted probabilities from XGBoost, LightGBM, and CatBoost and applies a validation-selected threshold. It is a simple average, without a separately trained meta-model.

The average has validation recall of 0.8187 and F1 of about 0.6583. It identifies more churn cases than some individual settings, with lower precision. Whether that tradeoff is useful would depend on campaign costs, capacity, and the expected benefit of an intervention.

## Customer segment summaries

The original report presents the following CatBoost test-cohort segment counts and observed churn rates. The groups can overlap.

| Segment | Customers | Observed churn rate |
| --- | ---: | ---: |
| Month-to-month customers | 1,350 | 0.5563 |
| New customers | 1,409 | 0.5358 |
| High monthly charge customers | 495 | 0.7374 |
| Low attachment customers | 966 | 0.5248 |

High monthly charge customers have the highest reported observed churn rate in this test cohort. Short contracts and limited service adoption also remain useful groups to investigate.

These are segment prevalence summaries. The supplied table does not contain segment precision, recall, F1, AUC, or confidence intervals, so it does not establish that the model performs equally well across segments. All test customers are in the new-customer segment under this split, which also limits comparisons between lifecycle groups.

## Evaluation measures

Accuracy describes overall correctness. Precision describes how many predicted churn cases are actual churn cases. Recall describes how many actual churn cases are found. F1 balances precision and recall. ROC-AUC measures ranking discrimination across thresholds. The confusion matrix shows correct non-churn predictions, false alarms, missed churn cases, and correctly identified churn cases.

In a retention setting, recall matters because missed churn cases represent missed opportunities to act, while precision matters because unnecessary outreach has a cost. No actual campaign outcome or financial return is measured here.

## Progress against the project targets

| Measure | Project target | Selected CatBoost validation result | Target reached? |
| --- | ---: | ---: | --- |
| ROC-AUC | 0.88 | 0.7999 | No |
| F1-score | 0.80 | 0.6586 | No |

Compared with the Week 5 baseline AUC of 0.7882 quoted in this source, CatBoost improves validation AUC by 0.0117. This is a modest gain, and the more ambitious performance targets were not reached.

The available static customer, contract, payment, and service variables may limit further gains. Behavioral histories, usage changes, valid linked support records, and payment failure information could be useful directions, but no improvement from those sources has been demonstrated here.

## Conclusion and next steps

The reported advanced models improve validation AUC slightly over the cited baseline. CatBoost leads on validation AUC, while LightGBM has the strongest advanced-model test F1 in the supplied table. Probability averaging offers a different precision-recall tradeoff rather than a clear overall improvement.

Next steps should begin with a stronger evaluation design: split before fitting preprocessing, use actual observation dates where available, keep the test set untouched during tuning, and compare full and selected feature sets. Additional tuning or weighted averaging could then be tested on validation data. Support features would require valid customer-level linkage before they could be added to the churn model.

The final repository includes the supplied training script in consolidated form and the original saved model bundle. A seeded new tuning run is recorded separately. The original trial history and original segment-performance CSV were not supplied; they are not reconstructed as historical evidence.
