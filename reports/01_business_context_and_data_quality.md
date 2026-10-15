# Business Context, Data Quality, and Project Roadmap

## Project background

This project looks at customer churn using customer characteristics, service subscriptions, contracts, payment methods, and charges. The aim is to understand which customers are more likely to leave and how those findings might support retention decisions.

The business context is inspired by cloud subscription services such as Microsoft Azure. In that setting, churn could mean cancelling a subscription, reducing usage, moving to another provider, or stopping purchases of cloud resources. Identifying customers at risk could help a business plan support and retention activities.

The analysis uses a Telco dataset as a proxy. These are not actual Azure customer records, and the findings have not been validated for Azure customers.

## Datasets and their roles

| Dataset | Original size | Role in the project |
| --- | --- | --- |
| Telco Customer Churn | 7,043 customers, 21 columns | Main dataset for churn analysis and prediction; includes the Churn label. |
| Customer Support Tickets | 8,469 tickets, 17 columns | Separate analysis of customer service experience; has no churn label. |
| SaaS churn benchmarks | Referenced in the original project plan | Industry context only; the benchmark workbook was not included in the supplied files. |

There is no shared customer identifier between the Telco and support datasets, so they are not merged.

## Telco overview

There are 5,174 non-churn customers and 1,869 churn customers. The overall churn rate is about 26.5%. The numerical summary below describes the data before filling the 11 blank TotalCharges values.

| Statistic | Tenure, months | MonthlyCharges | TotalCharges |
| --- | ---: | ---: | ---: |
| Non-missing count | 7,043 | 7,043 | 7,032 |
| Mean | 32.37 | 64.76 | 2,283.30 |
| Minimum | 0 | 18.25 | 18.80 |
| Median | 29 | 70.35 | 1,397.48 |
| Maximum | 72 | 118.75 | 8,684.80 |

The data covers both new and long-standing customers, with substantial differences in spending. The MonthlyCharges median and maximum above correct two apparent copy-and-paste errors in the original report, using the supplied data.

## Main business factors

### Contract type

| Contract | Churn rate |
| --- | ---: |
| Month-to-month | 42.71% |
| One year | 11.27% |
| Two year | 2.83% |

Customers on shorter contracts have higher observed churn. One possible explanation is that they face fewer barriers to cancelling. This makes contract type an important variable for later modeling, although the comparison does not establish a causal effect.

### Payment method

| Payment method | Churn rate |
| --- | ---: |
| Bank transfer, automatic | 16.71% |
| Credit card, automatic | 15.24% |
| Electronic check | 45.29% |
| Mailed check | 19.11% |

Automatic payment may be associated with more established usage and payment habits. Electronic check users have higher observed churn, but the data alone does not show that they are more price-sensitive.

### Technical support and internet service

| Technical support | Churn rate |
| --- | ---: |
| Yes | 15.17% |
| No | 41.64% |
| No internet service | 7.40% |

Customers without technical support have a much higher churn rate. Timely support may help customers use a product successfully, making this an interesting retention hypothesis for a cloud subscription business.

| Internet service | Churn rate |
| --- | ---: |
| DSL | 18.96% |
| Fiber optic | 41.89% |
| No internet service | 7.40% |

Customers using a more expensive service are not necessarily more stable. Higher expectations about quality, reliability, or value could help explain the pattern, but those explanations would need further evidence.

### Tenure and charges

Shorter-tenure customers may still be developing usage habits and may have lower switching costs. Longer-tenure customers may have more experience with the service and stronger reasons to stay.

MonthlyCharges represents current spending, while TotalCharges reflects accumulated spending and is also related to tenure. High monthly charges could indicate greater service use, greater price pressure, or both. Losing a customer with high accumulated spending may have a larger revenue impact, but these variables are not direct measures of future lifetime value.

### Security and backup services

| Service | Churn without the service | Churn with the service |
| --- | ---: | ---: |
| OnlineSecurity | 41.77% | 14.61% |
| OnlineBackup | 39.93% | 21.53% |

These differences suggest that service adoption and customer attachment are useful areas to explore. They do not prove that adding a service would itself prevent churn.

## Data quality assessment

### Telco data

The dataset has clear variable definitions and a usable churn label. The original checks found no duplicate rows. TotalCharges is stored as text in the source dataset; conversion to numeric reveals 11 blanks, corresponding to customers with zero tenure. The cleaning step fills these with zero.

Tenure, MonthlyCharges, and TotalCharges were checked using the 1.5 times IQR rule. Their observed ranges were reasonable for the dataset. Categories such as No internet service and No phone service have business meaning and are retained rather than treated as missing values.

### Support ticket data

The original missing-value counts were:

| Field | Missing records |
| --- | ---: |
| Resolution | 5,700 |
| First Response Time | 2,819 |
| Time to Resolution | 5,700 |
| Customer Satisfaction Rating | 5,700 |

These gaps are linked to ticket status. Closed tickets have resolution details, resolution timestamps, and satisfaction scores. Open and pending tickets do not yet have those final outcomes. First response timestamps are available for closed and pending tickets but missing for open tickets. This missingness should not be handled by simply filling every blank with an average.

No duplicate rows or duplicate Ticket IDs were found in the original checks. Customer ages range from 18 to 70, with an average of about 44.03. Satisfaction scores range from 1 to 5, with an average of about 2.99 among recorded scores. The reported gender counts are 2,896 male, 2,887 female, and 2,686 other.

| Ticket type | Count |
| --- | ---: |
| Refund request | 1,752 |
| Technical issue | 1,747 |
| Cancellation request | 1,695 |
| Product inquiry | 1,641 |
| Billing inquiry | 1,634 |

| Ticket status | Count |
| --- | ---: |
| Pending Customer Response | 2,881 |
| Open | 2,819 |
| Closed | 2,769 |

Priorities are fairly evenly distributed: Medium 2,192; Critical 2,129; High 2,085; Low 2,063. Channels are also fairly evenly distributed: Email 2,143; Phone 2,132; Social media 2,121; Chat 2,073. No obvious issue was identified in the ticket subject distribution.

A clear timestamp problem was found in 1,365 records: Time to Resolution is earlier than First Response Time. Before cleaning, the mean response-to-resolution interval is about -0.0577 hours and the minimum is about -23.23 hours. These records are flagged, and their derived resolution intervals are set to missing.

The time from purchase to first response is not the same as customer support response time. Ticket creation time is not available, so that interval must be interpreted as a purchase-to-response interval.

### SaaS benchmark reference

The original planning notes reference benchmark sheets for overall performance, customer segment, industry, company stage, and churn type. They quote B2B SaaS monthly churn of about 0.3%-1%, annual churn of about 3.5%-5%, and a broader gross monthly SaaS churn figure of about 3.5%. Those figures refer to different contexts and should not be treated as directly comparable measures.

The benchmark workbook and its original sources were not supplied, so these figures remain unverified background notes. No benchmark observations are used to train or evaluate the churn models. The intended takeaway was to explore whether customer size, contract length, company maturity, payment reliability, onboarding, and support could matter for retention.

## Planned eight-week roadmap

| Week | Focus | Planned output |
| --- | --- | --- |
| 1 | Business understanding and initial data checks | Business context, data quality review, roadmap, and at least eight potential churn factors. |
| 2 | Exploratory data analysis | Visual comparisons of churn by contract, payment, support, tenure, charges, and service use; initial hypotheses. |
| 3 | Data cleaning | Numeric conversion, missing-value handling, duplicate checks, and timestamp validation. |
| 4 | Feature engineering | Lifecycle, spending, service adoption, payment, and combined risk features. |
| 5 | Baseline modeling | Initial model comparison using accuracy, precision, recall, F1, and ROC-AUC. |
| 6 | Advanced modeling | Boosting models, tuning, and comparison with the baseline. |
| 7 | Interpretation and risk scoring | Main predictive factors and draft retention recommendations. |
| 8 | Customer segmentation | Customer groups based on estimated risk and value. |

This is the original roadmap. The complete repository now includes the Week 6 tuning code and saved model, Week 7 interpretation notebook and scorecard, and Week 8 segmentation code and outputs.

## Main project limitations

The Telco data is a proxy rather than Azure data. The datasets cannot be linked at customer level. Support timestamps contain inconsistencies. The churn classes are imbalanced. Some Telco services, such as streaming TV and movies, have limited relevance to cloud infrastructure. These limits should guide how the findings are interpreted and presented.
