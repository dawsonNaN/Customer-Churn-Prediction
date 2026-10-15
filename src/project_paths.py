"""Shared paths and local dataset loading for the portfolio package."""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data' / 'processed'
RAW_DIR = ROOT / 'data' / 'raw'

TELCO_COLUMNS = [
    'customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
    'MonthlyCharges', 'TotalCharges', 'Churn',
]
SUPPORT_COLUMNS = [
    'Ticket ID', 'Customer Age', 'Customer Gender', 'Product Purchased',
    'Date of Purchase', 'Ticket Type', 'Ticket Subject', 'Ticket Status',
    'Resolution', 'Ticket Priority', 'Ticket Channel', 'First Response Time',
    'Time to Resolution', 'Customer Satisfaction Rating',
]

def output_dir(stage):
    import os
    path = Path(os.environ.get('CHURN_OUTPUT_DIR', ROOT / 'outputs')) / stage
    path.mkdir(parents=True, exist_ok=True)
    return path

def load_telco_source():
    """Prefer a raw source file; otherwise reconstruct its 21-column view.

    The local fallback uses the supplied cleaned snapshot, restoring the 11
    zero-tenure TotalCharges values to missing for the initial quality checks.
    It is a reconstructed source view, not a newly downloaded raw dataset.
    """
    raw_path = RAW_DIR / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    if raw_path.exists():
        return pd.read_csv(raw_path)
    frame = pd.read_csv(DATA_DIR / 'telco_cleaned.csv')[TELCO_COLUMNS].copy()
    frame.loc[frame['tenure'].eq(0) & frame['TotalCharges'].eq(0), 'TotalCharges'] = float('nan')
    return frame

def load_support_source():
    """Load a raw file or the public copy and remove identity/free-text fields.

    Resolution retains only its missing/present information. No customer-level
    link to the Telco dataset is available.
    """
    raw_path = RAW_DIR / 'customer_support_tickets.csv'
    path = raw_path if raw_path.exists() else DATA_DIR / 'support_tickets_cleaned.csv'
    frame = pd.read_csv(path)
    if 'Ticket ID' not in frame:
        frame['Ticket ID'] = range(1, len(frame) + 1)
    frame = frame[[c for c in SUPPORT_COLUMNS if c in frame]].copy()
    frame['Resolution'] = frame['Resolution'].where(frame['Resolution'].isna(), 'Recorded')
    return frame


def lifecycle_split(frame, convert_booleans=False):
    """Reproduce the original tenure-descending 60/20/20 partition.

    This orders lifecycle groups; tenure is not a calendar observation date.
    """
    frame = frame.copy()
    target = 'Churn_binary'
    if target not in frame or 'tenure' not in frame:
        raise ValueError('The model matrix must contain tenure and Churn_binary.')
    if not frame[target].isin([0, 1]).all():
        raise ValueError('Churn_binary must contain only 0/1 values.')
    # The original baseline preserves boolean dummy dtypes, while Week 6
    # explicitly casts them to integers. Keep that distinction: it affects
    # how pandas-backed oversamplers restore generated column dtypes.
    if convert_booleans:
        for col in frame.select_dtypes(include=['bool']).columns:
            frame[col] = frame[col].astype(int)
    frame = frame.apply(pd.to_numeric, errors='raise')
    if frame.isna().any().any():
        raise ValueError('The model matrix contains missing values.')
    ordered = frame.sort_values('tenure', ascending=False).reset_index(drop=True)
    a, b = int(len(ordered) * .6), int(len(ordered) * .8)
    return {'Training': ordered.iloc[:a].copy(), 'Validation': ordered.iloc[a:b].copy(), 'Test': ordered.iloc[b:].copy()}


def classification_metrics(y_true, probabilities, threshold):
    """One metric implementation shared by training and interpretation stages."""
    import numpy as np
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
    predicted = (np.asarray(probabilities) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted, labels=[0, 1]).ravel()
    return {
        'accuracy': float(accuracy_score(y_true, predicted)),
        'precision': float(precision_score(y_true, predicted, zero_division=0)),
        'recall': float(recall_score(y_true, predicted, zero_division=0)),
        'f1_score': float(f1_score(y_true, predicted, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else float('nan'),
        'tn': int(tn), 'fp': int(fp), 'fn': int(fn), 'tp': int(tp),
    }


def best_f1_threshold(y_true, probabilities):
    import numpy as np
    from sklearn.metrics import f1_score
    thresholds = np.arange(.10, .91, .01)
    scores = [f1_score(y_true, np.asarray(probabilities) >= t, zero_division=0) for t in thresholds]
    return float(thresholds[int(np.argmax(scores))])
