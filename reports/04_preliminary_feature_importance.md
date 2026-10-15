# Preliminary Feature Importance Analysis

## Overview

Building on cleaning and exploratory analysis, this stage created a model-ready Telco feature set and designed a separate set of support experience features. The goal was to identify useful predictors and reduce repeated information before baseline modeling.

The Telco data is used for churn prediction because it has a churn label. Support features are not merged into that model because the datasets have no shared customer ID.

## Feature groups

### Service behavior

Features include has_tech_support, has_online_security, has_online_backup, has_device_protection, has_streaming_tv, has_streaming_movies, and has_multiple_lines. Aggregated indicators include support_service_count, add_on_service_count, is_basic_service_user, and has_multiple_services.

These variables describe the breadth of service adoption. Broader adoption may indicate stronger attachment to the provider. Basic-service customers may find it easier to switch, but this remains a business interpretation rather than a measured causal effect.

### Spending and value proxies

Features include MonthlyCharges, TotalCharges, avg_monthly_total_charge, charge_gap, charge_gap_ratio, is_high_monthly_charge, and is_high_total_charge. The high-charge flags use the relevant full-dataset median in the historical code.

These variables describe spending levels and the difference between current charges and average historical spending. They can be useful proxies for value or possible cost pressure, but they do not directly measure customer satisfaction or price sensitivity.

### Lifecycle

Tenure, tenure_group, is_new_customer, and is_long_term_customer represent stages of the customer relationship. The exploratory analysis suggested that early-stage customers had higher churn.

The original narrative also mentions is_mid_tenure_customer, but this field is not implemented in the supplied feature script and is not in the provided model matrix.

### Demographics

Features include SeniorCitizen, gender_Male, has_partner, has_dependents, and the initially constructed is_single_no_dependents. These add context about customer groups. In the original interpretation, they were secondary to contract, tenure, charges, and service adoption.

### Combined risk patterns

| Feature | Rule before scaling |
| --- | --- |
| high_risk_segment | Month-to-month contract, no TechSupport, and electronic check payment. |
| new_customer_no_support | New customer with has_tech_support = 0. |
| high_charge_no_support | High monthly charges with has_tech_support = 0. |
| month_to_month_high_charge | Month-to-month contract and high monthly charges. |
| low_attachment_customer | Month-to-month contract and at most one add-on service. |

Because has_tech_support maps only Yes to one, a zero also includes customers without internet service. This is broader than the explicit TechSupport = No rule used in high_risk_segment.

## Feature selection approaches

### Mutual information

Mutual information measures dependence between an individual feature and the churn label. The original report identifies is_month_to_month, tenure, month_to_month_high_charge, high_risk_segment, is_two_year_contract, uses_fiber_optic, is_new_customer, charge_gap, and is_electronic_check among the higher-ranked variables.

This suggests useful associations with contracts, lifecycle, charges, payments, and combined risk profiles. Mutual information does not establish direction or causality. The original script did not set a random seed inside mutual_info_classif. The consolidated preparation module sets it to 42 by default; newly generated rankings are recorded separately from the original report.

### Random Forest importance

A Random Forest with 200 trees and balanced class weights was used to rank features by impurity-based importance. The report highlights TotalCharges, MonthlyCharges, tenure, charge_gap, charge_gap_ratio, is_month_to_month, and month_to_month_high_charge.

This method can capture nonlinear relationships. Its importance scores may favor continuous variables and may distribute importance across correlated predictors, so they are not a direct estimate of business impact.

### Recursive feature elimination

Recursive feature elimination used balanced Logistic Regression as its base estimator and retained up to 30 predictors. Examples highlighted in the report include add_on_service_count, is_basic_service_user, has_multiple_services, uses_fiber_optic, has_dependents, low_attachment_customer, Contract_One year, and PaymentMethod_Credit card (automatic).

The script combines the top 30 mutual-information features, top 30 Random Forest features, and RFE-selected features using their union. This can retain more than 30 features. The full model matrix and selected-feature matrix are separate outputs.

## Correlation filtering

Absolute pairwise correlations above 0.90 were used to flag redundant columns. The feature count was reduced from 70 to 44 predictors, plus the target column.

The original report lists the following removed columns:

- avg_monthly_total_charge, is_long_contract, is_senior, is_single_no_dependents, new_customer_no_support
- Partner_Yes, Dependents_Yes, MultipleLines_No phone service, MultipleLines_Yes
- InternetService_Fiber optic, InternetService_No
- OnlineSecurity_No internet service, OnlineSecurity_Yes
- OnlineBackup_No internet service, OnlineBackup_Yes
- DeviceProtection_No internet service, DeviceProtection_Yes
- TechSupport_No internet service, TechSupport_Yes
- StreamingTV_No internet service, StreamingTV_Yes
- StreamingMovies_No internet service, StreamingMovies_Yes
- Contract_Two year, PaymentMethod_Electronic check, tenure_group_49-72 months

Filtering reduces pairwise redundancy. It does not prove that every form of multicollinearity has been eliminated. Correlations such as tenure with TotalCharges can still have a sensible business interpretation.

## Separate support feature design

Support features cover time intervals, ticket status, issue type, channel, priority, satisfaction, and combined warning signals. Examples include response_time_hours, resolution_time_hours, is_closed_ticket, is_technical_issue, is_billing_inquiry, is_refund_request, is_cancellation_request, is_low_satisfaction, and churn_warning_ticket.

Future customer-linked data could support features such as ticket counts, average satisfaction, unresolved issues, refund requests, cancellation requests, technical issue counts, and average valid resolution intervals. These were not validated as churn predictors in this project.

## Main takeaways

The preliminary analysis points to contracts, tenure, charges, payment method, and service adoption as useful prediction areas. Combined risk patterns also appear promising. Demographics provide supporting context.

These are exploratory feature findings from the full dataset. The ranking models were fitted on all available labels, and preprocessing was also performed before splitting. Later work should fit preprocessing and supervised feature selection only within the training data before assessing held-out performance.
