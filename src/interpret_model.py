"""CatBoost SHAP explanations, optional LIME, and a customer risk scorecard."""
import json
import hashlib
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from catboost import Pool
from project_paths import ROOT, DATA_DIR, output_dir, classification_metrics, load_telco_source


def run(data_path=None, model_path=None, seed=42, include_lime=True):
    data_path = data_path or DATA_DIR / 'telco_full_model_ready.csv'
    model_path = model_path or ROOT / 'models/week6_best_churn_model.pkl'
    frame = pd.read_csv(data_path).reset_index(drop=True)
    bundle = joblib.load(model_path)
    if bundle.get('model_name') != 'CatBoost' or 'model' not in bundle:
        raise ValueError('This interpretation stage requires a single CatBoost model bundle.')
    model, features, threshold = bundle['model'], bundle['features'], float(bundle['threshold'])
    if not 0 < threshold < 1:
        raise ValueError('The classification threshold must be between zero and one.')
    # Preserve the supplied model's 0.35 boundary. A newly tuned model can
    # select a lower decision threshold, so keep its bands non-overlapping.
    low_boundary = .35 if threshold > .35 else threshold / 2
    X = frame[features].astype(float)
    y = frame['Churn_binary']
    if len(frame) != len(load_telco_source()):
        raise ValueError('The interpretation/segmentation workflow expects the full original customer row order.')
    probabilities = model.predict_proba(X)[:, 1]
    risk = np.where(probabilities < low_boundary, 'Low Risk', np.where(probabilities < threshold, 'Medium Risk', 'High Risk'))
    # CatBoost native SHAP contributions are in the model's raw margin space.
    values = model.get_feature_importance(Pool(X), type='ShapValues', thread_count=2)
    contributions = values[:, :-1]
    global_importance = pd.DataFrame({'feature': features, 'mean_abs_shap_value': np.abs(contributions).mean(axis=0)}).sort_values('mean_abs_shap_value', ascending=False)
    ranks = np.argsort(-np.abs(contributions), axis=1, kind='stable')
    scorecard = pd.DataFrame({
        'customer_index': np.arange(len(frame)), 'churn_probability': probabilities,
        'risk_level': risk, 'actual_churn': y, 'predicted_churn': (probabilities >= threshold).astype(int),
    })
    for i in range(3): scorecard[f'top_risk_driver_{i+1}'] = np.asarray(features)[ranks[:, i]]
    summary = scorecard.groupby('risk_level').agg(n_customers=('customer_index', 'count'),
        avg_churn_probability=('churn_probability', 'mean'), actual_churn_rate=('actual_churn', 'mean')).reset_index()
    out = output_dir('interpretation')
    scorecard.to_csv(out / 'customer_risk_scorecard.csv', index=False)
    summary.to_csv(out / 'risk_level_summary.csv', index=False)
    global_importance.to_csv(out / 'shap_global_importance.csv', index=False)
    pd.DataFrame([{'scope': 'All customers; includes training observations', 'threshold': threshold,
        **classification_metrics(y, probabilities, threshold)}]).to_csv(out / 'full_sample_diagnostic_metrics.csv', index=False)
    local = []
    for row in np.argsort(-probabilities, kind='stable')[:5]:
        for col in ranks[row, :5]:
            local.append({'customer_index': int(row), 'churn_probability': probabilities[row],
                'risk_level': risk[row], 'feature': features[col], 'feature_value': X.iloc[row, col],
                'shap_value': contributions[row, col]})
    pd.DataFrame(local).to_csv(out / 'shap_local_explanations.csv', index=False)
    top = global_importance.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.5), layout='constrained')
    ax.barh(top.feature, top.mean_abs_shap_value, color='#444444')
    ax.set(xlabel='Mean absolute SHAP value (raw model margin)', title='Top 10 Features Explaining CatBoost Predictions')
    fig.savefig(out / 'shap_top10.png', dpi=180, bbox_inches='tight'); plt.close(fig)
    import shap
    shap.summary_plot(contributions, X, feature_names=features, show=False)
    plt.tight_layout(); plt.savefig(out / 'shap_summary.png', dpi=180, bbox_inches='tight'); plt.close('all')
    if include_lime:
        from lime.lime_tabular import LimeTabularExplainer
        # Avoid unregularized forward-selection fits on constant discretized
        # columns in this encoded matrix; retain LIME's regularized local fit.
        explainer = LimeTabularExplainer(X.to_numpy(), feature_names=features,
            class_names=['Non-Churn', 'Churn'], mode='classification',
            feature_selection='highest_weights', random_state=seed)
        lime_rows = []
        for row in np.argsort(-probabilities, kind='stable')[:3]:
            exp = explainer.explain_instance(X.iloc[row].to_numpy(), model.predict_proba, num_features=5)
            for rule, weight in exp.as_list():
                lime_rows.append({'customer_index': int(row), 'churn_probability': probabilities[row],
                    'risk_level': risk[row], 'lime_feature_rule': rule, 'lime_contribution': weight})
        pd.DataFrame(lime_rows).to_csv(out / 'lime_local_explanations.csv', index=False)
    row_ids = load_telco_source()['customerID'].astype(str).tolist()
    (out / 'scorecard_metadata.json').write_text(json.dumps({
        'model_name': 'CatBoost', 'high_risk_threshold': threshold, 'low_risk_boundary': low_boundary,
        'customer_count': len(frame), 'row_id_sha256': hashlib.sha256('\n'.join(row_ids).encode()).hexdigest(),
        'model_sha256': hashlib.sha256(model_path.read_bytes()).hexdigest(),
        'data_sha256': hashlib.sha256(data_path.read_bytes()).hexdigest(),
        'full_sample_metrics_are_held_out': False, 'shap_units': 'raw model margin',
        'top_driver_sort': 'absolute contribution; may increase or decrease the prediction',
        'lime_seed': seed, 'lime_computed': include_lime,
        'lime_feature_selection': 'highest_weights',
    }, indent=2) + '\n')
    print(f'Interpreted CatBoost for {len(frame)} customers; high-risk threshold: {threshold:.2f}')
    return out / 'customer_risk_scorecard.csv'
