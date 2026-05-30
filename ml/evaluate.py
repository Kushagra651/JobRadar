"""
ml/evaluate.py — Evaluate the trained XGBoost model.

Run:  python -m ml.evaluate
Output:
  - prints classification report + AUC
  - artifacts/roc_curve.png
  - artifacts/pr_curve.png
  - artifacts/calibration.png
"""

import pathlib
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    classification_report,
)
from sklearn.calibration import calibration_curve

FEATURES_PATH = pathlib.Path("data/features.parquet")
MODEL_PATH    = pathlib.Path("models/xgb_model.json")
ARTIFACT_DIR  = pathlib.Path("artifacts")
LABEL_COL     = "is_genuinely_senior"


def load_val_data():
    df = pd.read_parquet(FEATURES_PATH)
    X  = df.drop(columns=[LABEL_COL])
    y  = df[LABEL_COL]
    _, X_val, _, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    return X_val, y_val


def load_model() -> xgb.XGBClassifier:
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def plot_roc(y_true, y_proba, out: pathlib.Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc          = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve — Seniority Classifier")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  ROC-AUC: {auc:.4f}  →  {out}")


def plot_pr(y_true, y_proba, out: pathlib.Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    ap                   = average_precision_score(y_true, y_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Avg Precision: {ap:.4f}  →  {out}")


def plot_calibration(y_true, y_proba, out: pathlib.Path) -> None:
    fraction_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=10)
    plt.figure(figsize=(6, 5))
    plt.plot(mean_pred, fraction_pos, "s-", label="XGBoost")
    plt.plot([0, 1], [0, 1], "k--", label="Perfect")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Plot")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Calibration plot  →  {out}")


def evaluate() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    X_val, y_val = load_val_data()
    model        = load_model()
    y_proba      = model.predict_proba(X_val)[:, 1]
    y_pred       = (y_proba >= 0.5).astype(int)

    print("\n── Classification Report ──────────────────────────")
    print(classification_report(y_val, y_pred, target_names=["inflated", "genuine_senior"]))

    plot_roc(y_val, y_proba, ARTIFACT_DIR / "roc_curve.png")
    plot_pr (y_val, y_proba, ARTIFACT_DIR / "pr_curve.png")
    plot_calibration(y_val, y_proba, ARTIFACT_DIR / "calibration.png")


if __name__ == "__main__":
    evaluate()