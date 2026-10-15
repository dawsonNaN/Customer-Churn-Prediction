"""Logistic Regression and Decision Tree baselines under five imbalance settings."""
import json
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from project_paths import DATA_DIR, output_dir, lifecycle_split, classification_metrics


def run(data_path=None):
    path = data_path or DATA_DIR / 'telco_full_model_ready.csv'
    parts = lifecycle_split(pd.read_csv(path))
    train = parts['Training']
    X_train, y_train = train.drop(columns='Churn_binary'), train.Churn_binary
    methods = {
        'No Resampling': None, 'Class Weight Balanced': 'class_weight',
        'RandomOverSampler': RandomOverSampler(random_state=42),
        'SMOTE': SMOTE(random_state=42, k_neighbors=5),
        'ADASYN': ADASYN(random_state=42, n_neighbors=5),
    }
    out = output_dir('baselines')
    rows = []
    for method, sampler in methods.items():
        X, y = (X_train, y_train) if sampler is None or isinstance(sampler, str) else sampler.fit_resample(X_train, y_train)
        weight = 'balanced' if method == 'Class Weight Balanced' else None
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, solver='liblinear', random_state=42, class_weight=weight),
            'Decision Tree': DecisionTreeClassifier(max_depth=5, min_samples_leaf=30, random_state=42, class_weight=weight),
        }
        for name, model in models.items():
            model.fit(X, y)
            for partition in ['Validation', 'Test']:
                frame = parts[partition]
                p = model.predict_proba(frame.drop(columns='Churn_binary'))[:, 1]
                result = classification_metrics(frame.Churn_binary, p, .5)
                rows.append({'dataset': partition, 'model': name, 'imbalance_method': method, **result})
    metrics = pd.DataFrame(rows)
    ranking = metrics[metrics.dataset == 'Validation'].sort_values(['roc_auc', 'f1_score'], ascending=False)
    metrics.to_csv(out / 'baseline_model_metrics.csv', index=False)
    ranking.to_csv(out / 'baseline_validation_ranking.csv', index=False)
    (out / 'selected_setting.json').write_text(json.dumps(ranking.iloc[0].to_dict(), indent=2) + '\n')
    print('Baseline selected by validation AUC:', ranking.iloc[0]['model'], ranking.iloc[0]['imbalance_method'], round(ranking.iloc[0].roc_auc, 4))
    return metrics
