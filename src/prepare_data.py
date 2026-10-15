"""Clean both datasets and engineer their features in one preparation stage.

This combines the original four cleaning/feature scripts. Historical feature
definitions and full-dataset preprocessing are retained for reproduction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif, RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from project_paths import ROOT, load_telco_source, load_support_source, output_dir


def clean_telco(source=None):
    df = load_telco_source() if source is None else source.copy()
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df = df.drop_duplicates().copy()
    df['Churn_binary'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['tenure_group'] = pd.cut(df['tenure'], [-1, 12, 24, 48, 72],
        labels=['0-12 months', '13-24 months', '25-48 months', '49-72 months'])
    df['monthly_charge_group'] = pd.cut(df['MonthlyCharges'], [0, 35, 70, 90, 120],
        labels=['Low', 'Medium', 'High', 'Very High'])
    combined = ((df['Contract'] == 'Month-to-month') &
                (df['TechSupport'] == 'No') &
                (df['PaymentMethod'] == 'Electronic check'))
    df['high_risk_segment'] = combined.map({True: 'High Risk Segment', False: 'Other Customers'})
    df['is_new_customer'] = (df['tenure'] <= 12).astype(int)
    df['is_long_term_customer'] = (df['tenure'] >= 48).astype(int)
    df['avg_monthly_total_charge'] = df['TotalCharges'].div(df['tenure'].replace(0, np.nan)).fillna(0)
    df['charge_gap'] = df['MonthlyCharges'] - df['avg_monthly_total_charge']
    df['charge_gap_ratio'] = df['charge_gap'].div(df['avg_monthly_total_charge'].replace(0, np.nan)).fillna(0)
    return df


def engineer_telco(cleaned, out, seed=42):
    df = cleaned.copy()
    # These medians and the scaling/correlation steps intentionally reproduce
    # the original all-data preprocessing, rather than a leakage-free pipeline.
    df['is_high_monthly_charge'] = (df.MonthlyCharges > df.MonthlyCharges.median()).astype(int)
    df['is_high_total_charge'] = (df.TotalCharges > df.TotalCharges.median()).astype(int)
    df['is_month_to_month'] = (df.Contract == 'Month-to-month').astype(int)
    df['is_long_contract'] = df.Contract.isin(['One year', 'Two year']).astype(int)
    df['is_two_year_contract'] = (df.Contract == 'Two year').astype(int)
    df['is_electronic_check'] = (df.PaymentMethod == 'Electronic check').astype(int)
    df['is_auto_payment'] = df.PaymentMethod.isin(['Bank transfer (automatic)', 'Credit card (automatic)']).astype(int)
    service_map = {
        'has_tech_support': 'TechSupport', 'has_online_security': 'OnlineSecurity',
        'has_online_backup': 'OnlineBackup', 'has_device_protection': 'DeviceProtection',
    }
    for new, old in service_map.items(): df[new] = (df[old] == 'Yes').astype(int)
    df['support_service_count'] = df[list(service_map)].sum(axis=1)
    for new, old in {'has_streaming_tv': 'StreamingTV', 'has_streaming_movies': 'StreamingMovies',
                     'has_multiple_lines': 'MultipleLines'}.items():
        df[new] = (df[old] == 'Yes').astype(int)
    add_ons = list(service_map) + ['has_streaming_tv', 'has_streaming_movies', 'has_multiple_lines']
    df['add_on_service_count'] = df[add_ons].sum(axis=1)
    df['is_basic_service_user'] = (df.add_on_service_count <= 1).astype(int)
    df['has_multiple_services'] = (df.add_on_service_count >= 3).astype(int)
    df['has_internet_service'] = (df.InternetService != 'No').astype(int)
    df['uses_fiber_optic'] = (df.InternetService == 'Fiber optic').astype(int)
    df['is_senior'] = df.SeniorCitizen
    df['has_partner'] = (df.Partner == 'Yes').astype(int)
    df['has_dependents'] = (df.Dependents == 'Yes').astype(int)
    df['is_single_no_dependents'] = ((df.Partner == 'No') & (df.Dependents == 'No')).astype(int)
    df['high_risk_segment'] = ((df.Contract == 'Month-to-month') & (df.TechSupport == 'No') &
                               (df.PaymentMethod == 'Electronic check')).astype(int)
    df['new_customer_no_support'] = ((df.is_new_customer == 1) & (df.has_tech_support == 0)).astype(int)
    df['high_charge_no_support'] = ((df.is_high_monthly_charge == 1) & (df.has_tech_support == 0)).astype(int)
    df['month_to_month_high_charge'] = ((df.is_month_to_month == 1) & (df.is_high_monthly_charge == 1)).astype(int)
    df['low_attachment_customer'] = ((df.is_month_to_month == 1) & (df.add_on_service_count <= 1)).astype(int)
    encoded = pd.get_dummies(df.drop(columns=['customerID', 'Churn']), drop_first=True)
    y = encoded.pop('Churn_binary')
    numeric = encoded.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns
    encoded[numeric] = StandardScaler().fit_transform(encoded[numeric])
    corr = encoded.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    removed = [c for c in upper if (upper[c] > .90).any()]
    X = encoded.drop(columns=removed)
    k = min(30, X.shape[1])
    # Seed the previously unseeded MI ranking for a reproducible final edition.
    mi = pd.DataFrame({'feature': X.columns,
        'mutual_information_score': mutual_info_classif(X, y, random_state=seed)}).sort_values('mutual_information_score', ascending=False)
    rf = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=2).fit(X, y)
    rf_scores = pd.DataFrame({'feature': X.columns, 'random_forest_importance': rf.feature_importances_}).sort_values('random_forest_importance', ascending=False)
    rfe = RFE(LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear'), n_features_to_select=k).fit(X, y)
    rfe_scores = pd.DataFrame({'feature': X.columns, 'rfe_selected': rfe.support_, 'rfe_ranking': rfe.ranking_}).sort_values('rfe_ranking')
    chosen = set(mi.head(k).feature) | set(rf_scores.head(k).feature) | set(rfe_scores.loc[rfe_scores.rfe_selected, 'feature'])
    # Keep matrix order rather than the arbitrary order of a Python set.
    selected = [c for c in X if c in chosen]
    full = X.assign(Churn_binary=y.values)
    full.to_csv(out / 'telco_full_model_ready.csv', index=False)
    X[selected].assign(Churn_binary=y.values).to_csv(out / 'telco_selected_features_model_ready.csv', index=False)
    mi.to_csv(out / 'feature_selection_mutual_information.csv', index=False)
    rf_scores.to_csv(out / 'feature_selection_random_forest.csv', index=False)
    rfe_scores.to_csv(out / 'feature_selection_rfe_logistic_regression.csv', index=False)
    pd.DataFrame({'selected_feature': selected}).to_csv(out / 'final_selected_feature_list.csv', index=False)
    pd.DataFrame({'removed_feature': removed}).to_csv(out / 'removed_correlated_features.csv', index=False)
    return full


def clean_support(source=None):
    tickets = load_support_source() if source is None else source.copy()
    tickets = tickets.drop_duplicates().copy()
    for col in ['Date of Purchase', 'First Response Time', 'Time to Resolution']:
        tickets[col] = pd.to_datetime(tickets[col], errors='coerce')
    interval = tickets['First Response Time'] - tickets['Date of Purchase']
    tickets['days_since_purchase'] = interval.dt.days
    tickets['response_time_hours'] = interval.dt.total_seconds() / 3600
    tickets['resolution_time_hours'] = (tickets['Time to Resolution'] - tickets['First Response Time']).dt.total_seconds() / 3600
    invalid = tickets['Time to Resolution'].notna() & tickets['First Response Time'].notna() & (tickets['Time to Resolution'] < tickets['First Response Time'])
    tickets['is_invalid_time_record'] = invalid.astype(int)
    tickets['Invalid Time Record'] = invalid.map({True: 'Invalid', False: 'Valid'})
    tickets.loc[invalid, 'resolution_time_hours'] = np.nan
    tickets['is_closed_ticket'] = (tickets['Ticket Status'] == 'Closed').astype(int)
    # Missing intervals map to zero here, preserving the historical definition.
    tickets['is_long_resolution'] = (tickets.resolution_time_hours > 24).astype(int)
    tickets['purchase_month'] = tickets['Date of Purchase'].dt.month
    tickets['purchase_year'] = tickets['Date of Purchase'].dt.year
    tickets['purchase_day_of_week'] = tickets['Date of Purchase'].dt.dayofweek
    tickets['is_weekend_purchase'] = tickets.purchase_day_of_week.isin([5, 6]).astype(int)
    return tickets


def engineer_support(cleaned):
    df = cleaned.copy()
    groups = {
        'Ticket Status': {'is_closed_ticket': 'Closed', 'is_open_ticket': 'Open', 'is_pending_customer_response': 'Pending Customer Response'},
        'Ticket Type': {'is_technical_issue': 'Technical issue', 'is_billing_inquiry': 'Billing inquiry', 'is_refund_request': 'Refund request', 'is_cancellation_request': 'Cancellation request', 'is_product_inquiry': 'Product inquiry'},
        'Ticket Priority': {'is_low_priority': 'Low', 'is_medium_priority': 'Medium', 'is_high_priority': 'High', 'is_critical_priority': 'Critical'},
        'Ticket Channel': {'is_email_channel': 'Email', 'is_phone_channel': 'Phone', 'is_chat_channel': 'Chat', 'is_social_media_channel': 'Social media'},
    }
    df['has_resolution'] = df.Resolution.notna().astype(int)
    df['has_missing_satisfaction'] = df['Customer Satisfaction Rating'].isna().astype(int)
    for col, definitions in groups.items():
        for new, value in definitions.items(): df[new] = (df[col] == value).astype(int)
    df['is_high_or_critical_priority'] = df['Ticket Priority'].isin(['High', 'Critical']).astype(int)
    df['is_low_satisfaction'] = (df['Customer Satisfaction Rating'] <= 2).astype(int)
    df['is_high_satisfaction'] = (df['Customer Satisfaction Rating'] >= 4).astype(int)
    df['satisfaction_group'] = pd.cut(df['Customer Satisfaction Rating'], [0, 2, 3, 5], labels=['Low', 'Medium', 'High'])
    df['unresolved_high_priority'] = ((df.is_closed_ticket == 0) & (df.is_high_or_critical_priority == 1)).astype(int)
    df['low_satisfaction_long_resolution'] = ((df.is_low_satisfaction == 1) & (df.is_long_resolution == 1)).astype(int)
    df['technical_issue_long_resolution'] = ((df.is_technical_issue == 1) & (df.is_long_resolution == 1)).astype(int)
    df['billing_or_refund_issue'] = ((df.is_billing_inquiry == 1) | (df.is_refund_request == 1)).astype(int)
    df['churn_warning_ticket'] = ((df.is_cancellation_request == 1) | (df.is_refund_request == 1)).astype(int)
    return df


def run(seed=42):
    out = output_dir('preparation')
    telco = clean_telco()
    support = clean_support()
    telco.to_csv(out / 'telco_cleaned.csv', index=False)
    support.to_csv(out / 'support_tickets_cleaned.csv', index=False)
    matrix = engineer_telco(telco, out, seed)
    engineered_support = engineer_support(support)
    engineered_support.to_csv(out / 'support_tickets_feature_engineered.csv', index=False)
    for name in ['telco_feature_dictionary.csv', 'support_ticket_feature_dictionary.csv']:
        pd.read_csv(ROOT / 'docs' / name).to_csv(out / name, index=False)
    print(f'Prepared Telco: {telco.shape}; model matrix: {matrix.shape}; support: {engineered_support.shape}')
    return out / 'telco_full_model_ready.csv'
