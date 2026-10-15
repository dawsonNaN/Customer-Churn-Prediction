# Data Cleaning and Feature Engineering Design

## Purpose

This stage moves from exploratory analysis to cleaned data and feature preparation. Earlier work highlighted contract type, tenure, payment method, technical support, add-on services, charges, and support experience as areas worth investigating.

The Telco dataset remains the main modeling source because it contains the churn label. Support tickets are analyzed separately. SaaS benchmarks were planned as contextual references, not as customer-level training records.

## Cleaning workflow

### Telco data

TotalCharges is converted from text to numeric. This reveals 11 missing values for zero-tenure customers. The project fills those values with zero, keeping the customers rather than dropping them.

Duplicate rows are checked and removed if present. Tenure, MonthlyCharges, and TotalCharges are reviewed using their observed ranges and the 1.5 times IQR rule. The supplied data does not show obviously unreasonable numeric ranges.

No internet service and No phone service are retained as meaningful categories. They are different from missing values and should not be merged into a generic missing category.

### Support tickets

Date of Purchase, First Response Time, and Time to Resolution are converted to datetime. Missing resolution and satisfaction fields are retained because they are linked to ticket status.

Records with a resolution timestamp earlier than the first response are flagged with is_invalid_time_record. Their derived resolution_time_hours values are set to missing, while the remaining ticket information is retained.

The public working copy excludes customer names, email addresses, and ticket descriptions. Resolution text is replaced with a presence marker while keeping the original missing-value pattern. This preserves the has_resolution feature without publishing the free text.

### SaaS benchmark tables

The original plan proposed converting percentage fields for comparison and visualization. The workbook was not supplied, so no executable benchmark cleaning step is included.

## Encoding and scaling

The churn target is encoded as Churn_binary: Yes = 1 and No = 0. Categorical predictors are one-hot encoded for modeling. The historical feature script also standardizes numeric columns, including numeric binary indicators, while boolean dummy columns remain unscaled.

Support categories could be encoded in a future model with linked customer data. They are not added to the Telco model in this project.

The original feature matrix was prepared before the later train/validation/test split. Scaling, median thresholds, and correlation filtering therefore use the full dataset; feature ranking also uses all labels. This is a limitation of the historical workflow. A future evaluation should split first and fit all learned preprocessing and selection steps using training data only.

## Lifecycle and charge features

| Feature | Definition | Interpretation |
| --- | --- | --- |
| tenure_group | 0-12, 13-24, 25-48, or 49-72 months | Describes a customer's lifecycle stage. |
| is_new_customer | 1 when tenure <= 12 | Flags customers in the early lifecycle. |
| is_long_term_customer | 1 when tenure >= 48 | Flags longer-standing customers. The 48-month boundary differs from the final tenure-group boundary and is preserved from the code. |
| avg_monthly_total_charge | TotalCharges / tenure; zero when tenure = 0 | Historical average spending per month. |
| charge_gap | MonthlyCharges - avg_monthly_total_charge | Current charge relative to the historical average. |
| charge_gap_ratio | charge_gap / avg_monthly_total_charge; zero when the denominator is zero | Relative size of the charge difference. |

The charge gap variables are rough comparisons between current and cumulative spending. They are not observed month-by-month price changes.

## Support time and status features

| Feature | Definition |
| --- | --- |
| days_since_purchase | Whole days from purchase to first response. |
| response_time_hours | Hours from purchase to first response; not a ticket response-time SLA. |
| resolution_time_hours | Hours from first response to resolution, with reversed intervals set to missing. |
| is_invalid_time_record | 1 if resolution is earlier than first response and both timestamps exist. |
| is_closed_ticket | 1 when Ticket Status is Closed. |
| is_long_resolution | 1 when the available resolution interval exceeds 24 hours. |
| purchase_month | Month extracted from Date of Purchase. |
| purchase_year | Year extracted from Date of Purchase. |
| purchase_day_of_week | Monday = 0 through Sunday = 6. |
| is_weekend_purchase | 1 for Saturday or Sunday. |

The original design document repeats purchase_day_of_week in its last item. It is listed once here, alongside the weekend indicator that is implemented in the code.

## Interpretation limits

Without a ticket creation timestamp, response_time_hours cannot measure how long support took to answer a submitted ticket. Missing timestamps are not proof of a valid interval. The original boolean comparisons encode missing intervals as zero for is_long_resolution; missing satisfaction similarly becomes zero in low/high satisfaction flags, so the separate missingness flags are important.

In the supplied cleaned data, no valid response-to-resolution interval exceeds 24 hours. The long-resolution flag is therefore constant and currently adds no variation. Support-derived risk flags remain ideas for future linked data rather than demonstrated churn predictors.
