"""
evaluate_svm.py - Evaluation script for Capstone-Website SVM.

Loads model directly from model_auth.pkl (no retraining needed).
Run svm.py once first to generate model_auth.pkl, then run this.

Usage (from inside Capstone-Website/):
    python evaluate_svm.py
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import cv2
from skimage.morphology import skeletonize

_GENUINE_DIR = "data/genuine"
_FORGED_DIR  = "data/forged"
_MODEL_FILE  = "model_auth.pkl"


def _get_features(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist  = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    sw    = dist[binary > 0]
    sw_cv = float(sw.std() / sw.mean()) if len(sw) > 0 and sw.mean() > 0 else 0.0
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    scores = []
    for cnt in contours:
        if len(cnt) > 10:
            pts    = cnt.reshape(-1, 2).astype(float)
            diffs  = np.diff(pts, axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            chg    = np.abs(np.diff(angles))
            chg    = np.minimum(chg, 2 * np.pi - chg)
            scores.append(float(np.mean(chg)))
    smoothness = float(np.mean(scores)) if scores else 0.5
    n_comp  = len(contours)
    ink     = float(np.sum(binary > 0)) / binary.size
    skeleton = skeletonize(binary > 0)
    retrace  = float(np.sum(skeleton)) / (float(np.sum(binary > 0)) + 1e-10)
    coords  = cv2.findNonZero(binary)
    if coords is not None:
        xs     = coords[:, 0, 0]
        ys     = coords[:, 0, 1]
        aspect = float(xs.max() - xs.min()) / float(ys.max() - ys.min() + 1e-10)
    else:
        aspect = 1.0
    return [sw_cv, smoothness, n_comp, ink, retrace, aspect]


def load_dataset():
    X, y = [], []
    for name in sorted(os.listdir(_GENUINE_DIR)):
        try:
            X.append(_get_features(f"{_GENUINE_DIR}/{name}"))
            y.append(1)
        except Exception:
            pass
    for name in sorted(os.listdir(_FORGED_DIR)):
        try:
            X.append(_get_features(f"{_FORGED_DIR}/{name}"))
            y.append(0)
        except Exception:
            pass
    return np.array(X, dtype=np.float32), np.array(y, dtype=int)


def main():
    print("=" * 60)
    print("  eSIGNIFY - Capstone SVM Evaluation")
    print("=" * 60)

    if not os.path.exists(_MODEL_FILE):
        print(f"\n[ERROR] {_MODEL_FILE} not found.")
        print("  Run svm.py first:  python svm.py")
        return

    with open(_MODEL_FILE, "rb") as f:
        payload = pickle.load(f)

    model  = payload["model"]
    scaler = payload["scaler"]
    print(f"\nLoaded model from {_MODEL_FILE}")
    print(f"  Trained on: {payload['n_genuine']} genuine | {payload['n_forged']} forged")

    print("\nExtracting features from dataset...")
    X, y = load_dataset()
    print(f"  Total: {len(y)}  ({int(np.sum(y==1))} genuine / {int(np.sum(y==0))} forged)")

    X_sc    = scaler.transform(X)
    y_pred  = model.predict(X_sc)
    y_proba = model.predict_proba(X_sc)[:, 1]

    acc  = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, zero_division=0)
    rec  = recall_score(y, y_pred, zero_division=0)
    f1   = f1_score(y, y_pred, zero_division=0)
    roc  = roc_auc_score(y, y_proba)

    print("\n--- Metrics (full dataset via saved model) ---------------")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {roc:.4f}")

    print("\n--- Classification Report --------------------------------")
    print(classification_report(y, y_pred, target_names=["Forged (0)", "Genuine (1)"]))

    print("--- 5-Fold Cross-Validation ------------------------------")
    cv        = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_sc, y, cv=cv, scoring='accuracy')
    print(f"  Folds : {cv_scores.round(4)}")
    print(f"  Mean  : {cv_scores.mean():.4f}  +/-  {cv_scores.std():.4f}")

    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("eSIGNIFY - Capstone SVM Evaluation", fontsize=14, fontweight='bold')

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=["Forged", "Genuine"],
                yticklabels=["Forged", "Genuine"],
                ax=axes[0])
    axes[0].set_xlabel("Predicted", fontsize=12)
    axes[0].set_ylabel("Actual", fontsize=12)
    axes[0].set_title("Confusion Matrix")
    axes[0].text(0.5, -0.18, f"TN={tn}  FP={fp}  FN={fn}  TP={tp}",
                 transform=axes[0].transAxes, ha='center', fontsize=10)

    RocCurveDisplay.from_predictions(y, y_proba, ax=axes[1],
                                      name=f"SVM  AUC={roc:.3f}")
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1)
    axes[1].set_title("ROC Curve")

    plt.tight_layout()
    plt.savefig("svm_evaluation_results.png", dpi=150, bbox_inches='tight')
    print("\nPlot saved -> svm_evaluation_results.png")

    print("\n--- Summary ----------------------------------------------")
    print(f"  TP (genuine  -> genuine) : {tp}")
    print(f"  TN (forged   -> forged)  : {tn}")
    print(f"  FP (forged   -> genuine) : {fp}")
    print(f"  FN (genuine  -> forged)  : {fn}")
    print("=" * 60)


if __name__ == "__main__":
    main()