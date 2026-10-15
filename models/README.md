# Saved Model

week6_best_churn_model.pkl is an unchanged copy of the supplied fitted CatBoost bundle. It contains the model, the 44 predictor names, and the approximately 0.64 classification threshold. supplied_model_metadata.json records its hash, model settings, and verification context.

Newly trained models are written to outputs/advanced_models. They do not replace this supplied model automatically. The full pipeline explains its newly trained CatBoost; the standalone interpret command uses this supplied bundle by default.
