"""
ml/predict.py — Load trained model and run inference.

This is the ONLY file analyze_node.py should import from the ml/ package.
It never does any MLflow logging — that lives entirely in train.py.
"""

import json
import pathlib
import pandas as pd
import xgboost as xgb

from ml.explain import explain_single

import config.settings as settings
MODEL_PATH   = pathlib.Path(settings.MODEL_PATH)
META_PATH    = pathlib.Path(settings.MODEL_PATH).parent / "feature_meta.json"

# Load once at module import — avoids reloading on every agent call
_model: xgb.XGBClassifier | None = None
_feature_names: list[str] | None = None


def _load() -> None:
    global _model, _feature_names
    if _model is None:
        _model = xgb.XGBClassifier()
        _model.load_model(MODEL_PATH)
        _feature_names = json.loads(META_PATH.read_text())["feature_names"]


def predict(features: dict) -> dict:
    """
    Args:
        features: flat dict of {feature_name: value} matching training schema

    Returns:
        {
          "is_senior": bool,
          "probability": float,
          "shap_values": {feature: shap_value, ...}
        }
    """
    _load()
    row      = pd.DataFrame([features])[_feature_names]   # enforce column order
    proba    = float(_model.predict_proba(row)[0][1])
    shap_map = explain_single(_model, row)

    return {
        "is_senior"  : proba >= 0.5,
        "probability": round(proba, 4),
        "shap_values": shap_map,
    }