"""Advanced tuning consolidated from the original Week 6 script.

Search ranges and model definitions are retained. Optuna is now seeded,
thread counts are bounded, and ranking uses unrounded validation AUC.
"""
import json
import numpy as np
import pandas as pd
import joblib
import optuna
from sklearn.metrics import f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from project_paths import DATA_DIR, output_dir, lifecycle_split, classification_metrics, best_f1_threshold

def run(data_path=None, trials=10, seed=42):
    file_path = data_path or DATA_DIR / 'telco_full_model_ready.csv'
    parts = lifecycle_split(pd.read_csv(file_path), convert_booleans=True)
    train_df, valid_df, test_df = parts['Training'], parts['Validation'], parts['Test']
    target_col = 'Churn_binary'
    X_train, y_train = train_df.drop(columns=target_col), train_df[target_col]
    X_valid, y_valid = valid_df.drop(columns=target_col), valid_df[target_col]
    X_test, y_test = test_df.drop(columns=target_col), test_df[target_col]
    OUT = output_dir('advanced_models')

    print("Dataset split completed.")
    print("Training set:", train_df.shape)
    print("Validation set:", valid_df.shape)
    print("Test set:", test_df.shape)

    print("\nChurn rate by split:")
    print("Train:", round(y_train.mean(), 4))
    print("Validation:", round(y_valid.mean(), 4))
    print("Test:", round(y_test.mean(), 4))


    # ============================================================
    # 3. Helper Functions
    # ============================================================

    def find_best_threshold(y_true, y_prob):
        threshold = best_f1_threshold(y_true, y_prob)
        return threshold, f1_score(y_true, np.asarray(y_prob) >= threshold, zero_division=0)

    def evaluate_model(model_name, y_true, y_prob, threshold, dataset_name):
        result = classification_metrics(y_true, y_prob, threshold)
        result['auc_roc'] = result.pop('roc_auc')
        return {'dataset': dataset_name, 'model': model_name, 'threshold': threshold, **result}

    # ============================================================
    # 4. Class Imbalance Setting
    # ============================================================
    # For boosting models, class imbalance can be handled by class weight.
    # scale_pos_weight = number of non-churn customers / number of churn customers

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    print("\nClass imbalance setting:")
    print("scale_pos_weight:", round(scale_pos_weight, 4))


    # ============================================================
    # 5. Optuna Hyperparameter Tuning
    # ============================================================

    N_TRIALS = trials
    optuna.logging.set_verbosity(optuna.logging.WARNING)


    def objective_xgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350),
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "subsample": trial.suggest_float("subsample", 0.65, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.00),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.10, 10.00, log=True),
            "scale_pos_weight": scale_pos_weight,
            "eval_metric": "logloss",
            "random_state": 42,
            "n_jobs": 2
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, valid_prob)


    def objective_lgb(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 350),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "subsample": trial.suggest_float("subsample", 0.65, 1.00),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.00),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 80),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.10, 10.00, log=True),
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": 2,
            "verbose": -1
        }

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, valid_prob)


    def objective_cat(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 100, 350),
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.20, log=True),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.00, 10.00, log=True),
            "random_strength": trial.suggest_float("random_strength", 0.10, 5.00),
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "auto_class_weights": "Balanced",
            "random_seed": 42,
            "verbose": False, "thread_count": 2, "allow_writing_files": False
        }

        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train)

        valid_prob = model.predict_proba(X_valid)[:, 1]
        return roc_auc_score(y_valid, valid_prob)


    studies = {}

    for model_name, objective_function in [
        ("XGBoost", objective_xgb),
        ("LightGBM", objective_lgb),
        ("CatBoost", objective_cat)
    ]:
        print("\nTuning", model_name)
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
        study.optimize(objective_function, n_trials=N_TRIALS, show_progress_bar=False)

        studies[model_name] = study

        print("Best validation AUC:", round(study.best_value, 4))
        print("Best parameters:", study.best_params)


    # ============================================================
    # 6. Train Final Tuned Models
    # ============================================================

    models = {}

    xgb_params = studies["XGBoost"].best_params.copy()
    xgb_params.update({
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": 2
    })
    models["XGBoost"] = XGBClassifier(**xgb_params)
    models["XGBoost"].fit(X_train, y_train)


    lgb_params = studies["LightGBM"].best_params.copy()
    lgb_params.update({
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": 2,
        "verbose": -1
    })
    models["LightGBM"] = LGBMClassifier(**lgb_params)
    models["LightGBM"].fit(X_train, y_train)


    cat_params = studies["CatBoost"].best_params.copy()
    cat_params.update({
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "auto_class_weights": "Balanced",
        "random_seed": 42,
        "verbose": False, "thread_count": 2, "allow_writing_files": False
    })
    models["CatBoost"] = CatBoostClassifier(**cat_params)
    models["CatBoost"].fit(X_train, y_train)


    # ============================================================
    # 7. Evaluate Individual Models
    # ============================================================

    results = []
    valid_probs = {}
    test_probs = {}
    thresholds = {}

    for model_name, model in models.items():
        valid_prob = model.predict_proba(X_valid)[:, 1]
        test_prob = model.predict_proba(X_test)[:, 1]

        best_threshold, best_valid_f1 = find_best_threshold(y_valid, valid_prob)

        valid_probs[model_name] = valid_prob
        test_probs[model_name] = test_prob
        thresholds[model_name] = best_threshold

        results.append(evaluate_model(model_name, y_valid, valid_prob, best_threshold, "Validation"))
        results.append(evaluate_model(model_name, y_test, test_prob, best_threshold, "Test"))


    # ============================================================
    # 8. Blending Model
    # ============================================================
    # Simple blending: average the predicted probabilities of XGBoost,
    # LightGBM, and CatBoost.

    blend_valid_prob = np.mean(
        np.column_stack([valid_probs[name] for name in models.keys()]),
        axis=1
    )

    blend_test_prob = np.mean(
        np.column_stack([test_probs[name] for name in models.keys()]),
        axis=1
    )

    blend_threshold, blend_valid_f1 = find_best_threshold(y_valid, blend_valid_prob)

    results.append(evaluate_model("Blending Average", y_valid, blend_valid_prob, blend_threshold, "Validation"))
    results.append(evaluate_model("Blending Average", y_test, blend_test_prob, blend_threshold, "Test"))


    performance_df = pd.DataFrame(results)

    metric_cols = ["accuracy", "precision", "recall", "f1_score", "auc_roc"]
    # Keep full precision for model ranking and exported metrics.

    performance_df.to_csv(OUT / "week6_advanced_model_performance.csv", index=False)

    print("\nAdvanced model performance:")
    print(performance_df)


    # ============================================================
    # 9. Select Best Model by Validation AUC
    # ============================================================

    validation_ranking = performance_df[performance_df["dataset"] == "Validation"].sort_values(
        by="auc_roc",
        ascending=False
    )

    validation_ranking.to_csv(OUT / "week6_validation_model_ranking.csv", index=False)

    best_model_name = validation_ranking.iloc[0]["model"]
    best_threshold = validation_ranking.iloc[0]["threshold"]

    print("\nBest model based on validation AUC:")
    print(best_model_name)
    print("Best threshold:", best_threshold)


    # ============================================================
    # 10. Segment Performance Evaluation
    # ============================================================

    if best_model_name == "Blending Average":
        best_test_prob = blend_test_prob
    else:
        best_test_prob = test_probs[best_model_name]

    segments = {
        "Month-to-month customers": X_test["is_month_to_month"] > 0,
        "New customers": X_test["is_new_customer"] > 0,
        "High monthly charge customers": X_test["is_high_monthly_charge"] > 0,
        "Low attachment customers": X_test["low_attachment_customer"] > 0
    }

    segment_results = []

    for segment_name, segment_mask in segments.items():
        segment_mask = np.array(segment_mask)

        if segment_mask.sum() > 10 and len(np.unique(y_test.values[segment_mask])) > 1:
            row = evaluate_model(
                best_model_name,
                y_test.values[segment_mask],
                best_test_prob[segment_mask],
                best_threshold,
                segment_name
            )

            row["n_customers"] = int(segment_mask.sum())
            row["churn_rate"] = float(y_test.values[segment_mask].mean())

            segment_results.append(row)

    segment_performance_df = pd.DataFrame(segment_results)

    if not segment_performance_df.empty:
        segment_performance_df[metric_cols + ["churn_rate"]] = segment_performance_df[
            metric_cols + ["churn_rate"]
        ].round(4)

    segment_performance_df.to_csv(OUT / "week6_segment_performance.csv", index=False)

    print("\nSegment performance:")
    print(segment_performance_df)


    # ============================================================
    # 11. Save Optuna Tuning Summary
    # ============================================================

    tuning_summary = []

    for model_name, study in studies.items():
        tuning_summary.append({
            "model": model_name,
            "best_validation_auc": study.best_value,
            "best_params": json.dumps(study.best_params)
        })

    tuning_summary_df = pd.DataFrame(tuning_summary)
    tuning_summary_df.to_csv(OUT / "week6_optuna_tuning_summary.csv", index=False)


    # ============================================================
    # 12. Save Best Model
    # ============================================================

    if best_model_name == "Blending Average":
        model_object = {
            "model_name": best_model_name,
            "models": models,
            "threshold": best_threshold,
            "features": X_train.columns.tolist()
        }
    else:
        model_object = {
            "model_name": best_model_name,
            "model": models[best_model_name],
            "threshold": best_threshold,
            "features": X_train.columns.tolist()
        }

    joblib.dump(model_object, OUT / "week6_best_churn_model.pkl")


    print("1. week6_advanced_model_performance.csv")
    print("2. week6_validation_model_ranking.csv")
    print("4. week6_optuna_tuning_summary.csv")
    print("5. week6_best_churn_model.pkl")

    # Save an explicit CatBoost explanation bundle even if another model wins AUC.
    cat_bundle = {'model_name': 'CatBoost', 'model': models['CatBoost'],
                  'threshold': thresholds['CatBoost'], 'features': X_train.columns.tolist()}
    joblib.dump(cat_bundle, OUT / 'catboost_explanation_model.pkl')
    for name, study in studies.items():
        study.trials_dataframe().to_csv(OUT / (name.lower() + '_optuna_trials.csv'), index=False)
    return {'best_bundle': OUT / 'week6_best_churn_model.pkl',
            'catboost_bundle': OUT / 'catboost_explanation_model.pkl',
            'best_model_name': best_model_name}
