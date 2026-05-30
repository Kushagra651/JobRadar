"""
ml/train.py — Train XGBoost seniority classifier.

Run:  python -m ml.train
Output:
  - MLflow run with params, metrics, model artifact
  - models/xgb_model.json  (local, for fast loading at inference)
"""

import os
import json
import pathlib
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from config.settings import settings

# ── paths ──────────────────────────────────────────────────────────────────────
FEATURES_PATH = pathlib.Path("data/features.parquet")
MODEL_DIR     = pathlib.Path("models")
MODEL_PATH    = MODEL_DIR / "xgb_model.json"
LABEL_COL     = "is_genuinely_senior"

# ── helpers ────────────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_parquet(FEATURES_PATH)
    X  = df.drop(columns=[LABEL_COL])
    y  = df[LABEL_COL]
    return X, y


def build_model() -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators     = 300,
        max_depth        = 6,
        learning_rate    = 0.05,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = 42,
        n_jobs           = -1,
    )


# ── main ───────────────────────────────────────────────────────────────────────
def train() -> None:
    # Set MLflow tracking URI from settings (falls back to local mlruns/)
    mlflow.set_tracking_uri(getattr(settings, "MLFLOW_TRACKING_URI", "mlruns"))
    mlflow.set_experiment("jobradar-seniority")

    X, y = load_data()
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Positive rate: {y.mean():.2%}")

    model = build_model()

    with mlflow.start_run(run_name="xgb-baseline"):
        # log hyper-params
        mlflow.log_params(model.get_params())

        # fit
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50,
        )

        # evaluate
        val_proba = model.predict_proba(X_val)[:, 1]
        auc       = roc_auc_score(y_val, val_proba)
        print(f"Validation ROC-AUC: {auc:.4f}")

        if auc < 0.78:
            print("⚠  AUC below 0.78 target — review features before deploying")

        mlflow.log_metric("val_roc_auc", auc)

        # log model to MLflow artifact store
        mlflow.xgboost.log_model(model, artifact_path="model")

        # also save locally so analyze_node can load without MLflow at runtime
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(MODEL_PATH)
        mlflow.log_artifact(str(MODEL_PATH))          # keep a copy in MLflow too

        # persist feature column order — critical for inference
        feature_meta = {"feature_names": list(X_train.columns)}
        meta_path    = MODEL_DIR / "feature_meta.json"
        meta_path.write_text(json.dumps(feature_meta, indent=2))
        mlflow.log_artifact(str(meta_path))

        print(f"✓ Model saved → {MODEL_PATH}")
        print(f"✓ MLflow run logged (experiment: jobradar-seniority)")


if __name__ == "__main__":
    train()