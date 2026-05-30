"""
ml/explain.py — SHAP explainability for the trained model.

Run:  python -m ml.explain
Output:
  - artifacts/shap_summary.png    (global feature importance)
  - artifacts/shap_force_<i>.png  (individual prediction explanation, first 3 val samples)
"""

import pathlib
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import config.settings as settings
FEATURES_PATH = pathlib.Path(settings.FEATURES_DATA_PATH)
MODEL_PATH    = pathlib.Path(settings.MODEL_PATH)
ARTIFACT_DIR  = pathlib.Path("artifacts")
LABEL_COL     = "is_genuinely_senior"
N_FORCE_PLOTS = 3   # individual explanations to save


def load_val_data():
    df = pd.read_parquet(FEATURES_PATH)
    X  = df.drop(columns=[LABEL_COL])
    y  = df[LABEL_COL]
    _, X_val, _, _ = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_val


def load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def explain() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_val = load_val_data()
    model = load_model()

    # TreeExplainer is native to XGBoost — fast and exact
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_val)

    # ── Summary plot (global) ──────────────────────────────────────────────────
    shap.summary_plot(shap_values, X_val, show=False)
    plt.tight_layout()
    summary_path = ARTIFACT_DIR / "shap_summary.png"
    plt.savefig(summary_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"✓ SHAP summary  →  {summary_path}")

    # ── Force plots (individual) ───────────────────────────────────────────────
    # Use matplotlib=True so we can save to file (JS plots don't save as PNG)
    shap.initjs()
    for i in range(min(N_FORCE_PLOTS, len(X_val))):
        force_path = ARTIFACT_DIR / f"shap_force_{i}.png"
        shap.force_plot(
            explainer.expected_value,
            shap_values[i],
            X_val.iloc[i],
            matplotlib=True,
            show=False,
        )
        plt.savefig(force_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"✓ SHAP force {i}  →  {force_path}")


# ── Public helper called by analyze_node at inference time ─────────────────────
def explain_single(model: xgb.XGBClassifier, feature_row: pd.DataFrame) -> dict:
    """
    Returns a dict of {feature_name: shap_value} for a single prediction.
    analyze_node calls this to attach per-prediction explanations to AgentState.
    """
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(feature_row)
    return dict(zip(feature_row.columns, shap_values[0].tolist()))


if __name__ == "__main__":
    explain()