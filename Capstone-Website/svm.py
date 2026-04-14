"""
svm.py - Signature authenticity detector (Capstone-Website).

KEY IMPROVEMENT over the old version:
  - Trains once and saves the model to model_auth.pkl
  - On subsequent runs, loads from pkl instantly (no retraining)
  - Output is DETERMINISTIC — same image always gives same result
  - Falls back to retraining automatically if pkl is missing or stale

Features used (6):
  1. Stroke-width variation  — natural pressure changes
  2. Stroke smoothness       — flow vs hesitation
  3. Pen-lift count          — natural breaks
  4. Ink coverage            — how much of the frame is filled
  5. Retrace ratio           — skeleton vs ink area
  6. Aspect ratio            — width-to-height of bounding box

Threshold: 0.55 (genuine probability) — optimal from training data (~94.5% acc)
"""

import os
import pickle
import hashlib
import numpy as np
import cv2
from skimage.morphology import skeletonize
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from os import listdir

# ── Paths ─────────────────────────────────────────────────────────────────────
_GENUINE_DIR  = "data/genuine"
_FORGED_DIR   = "data/forged"
_MODEL_FILE   = "model_auth.pkl"   # separate from the old SIFT model.pkl

# ── Module-level cache ────────────────────────────────────────────────────────
_model  = None
_scaler = None


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction
# ─────────────────────────────────────────────────────────────────────────────

def _get_features(path):
    """Extract 6 authenticity features from a signature image."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read: {path}")

    _, binary = cv2.threshold(img, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 1. Stroke-width variation (pressure naturalness)
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    sw   = dist[binary > 0]
    sw_cv = float(sw.std() / sw.mean()) if len(sw) > 0 and sw.mean() > 0 else 0.0

    # 2. Stroke smoothness (flow vs hesitation)
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

    # 3. Pen-lift count (connected components)
    n_comp = len(contours)

    # 4. Ink coverage
    ink = float(np.sum(binary > 0)) / binary.size

    # 5. Retrace ratio (skeleton / ink)
    skeleton = skeletonize(binary > 0)
    retrace  = float(np.sum(skeleton)) / (float(np.sum(binary > 0)) + 1e-10)

    # 6. Aspect ratio
    coords = cv2.findNonZero(binary)
    if coords is not None:
        xs     = coords[:, 0, 0]
        ys     = coords[:, 0, 1]
        aspect = float(xs.max() - xs.min()) / float(ys.max() - ys.min() + 1e-10)
    else:
        aspect = 1.0

    return [sw_cv, smoothness, n_comp, ink, retrace, aspect]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset Fingerprint  (used to detect if training data changed)
# ─────────────────────────────────────────────────────────────────────────────

def _dataset_fingerprint():
    """
    Returns a short hash of the filenames + modification times in the
    data directories.  If new images are added the fingerprint changes
    and the model gets retrained automatically.
    """
    items = []
    for d in [_GENUINE_DIR, _FORGED_DIR]:
        if os.path.isdir(d):
            for f in sorted(os.listdir(d)):
                fp = os.path.join(d, f)
                items.append(f"{fp}:{os.path.getmtime(fp)}")
    raw = "\n".join(items).encode()
    return hashlib.md5(raw).hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Model Training  (only runs when needed)
# ─────────────────────────────────────────────────────────────────────────────

def _train_and_save():
    """Train SVC on all available data and persist to model_auth.pkl."""
    X, y = [], []
    skipped = 0

    print("[SVM] Training model from scratch...")

    for name in sorted(os.listdir(_GENUINE_DIR)):
        try:
            X.append(_get_features(f"{_GENUINE_DIR}/{name}"))
            y.append(1)
        except Exception as e:
            print(f"  [skip genuine] {name}: {e}")
            skipped += 1

    for name in sorted(os.listdir(_FORGED_DIR)):
        try:
            X.append(_get_features(f"{_FORGED_DIR}/{name}"))
            y.append(0)
        except Exception as e:
            print(f"  [skip forged] {name}: {e}")
            skipped += 1

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=int)

    scaler = StandardScaler().fit(X)
    X_sc   = scaler.transform(X)
    model  = SVC(kernel='rbf', C=10, gamma='scale',
                 probability=True, random_state=42)
    model.fit(X_sc, y)

    fingerprint = _dataset_fingerprint()

    payload = {
        "model":       model,
        "scaler":      scaler,
        "fingerprint": fingerprint,
        "n_genuine":   int(np.sum(y == 1)),
        "n_forged":    int(np.sum(y == 0)),
    }
    with open(_MODEL_FILE, "wb") as f:
        pickle.dump(payload, f)

    print(f"[SVM] Model saved → {_MODEL_FILE}")
    print(f"      {payload['n_genuine']} genuine | {payload['n_forged']} forged "
          f"| {skipped} skipped")

    return model, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Model Loading  (with staleness check)
# ─────────────────────────────────────────────────────────────────────────────

def _load_model():
    """
    Load model from pkl if it exists and training data hasn't changed.
    Otherwise retrain.
    """
    global _model, _scaler

    current_fp = _dataset_fingerprint()

    if os.path.exists(_MODEL_FILE):
        try:
            with open(_MODEL_FILE, "rb") as f:
                payload = pickle.load(f)

            if payload.get("fingerprint") == current_fp:
                _model  = payload["model"]
                _scaler = payload["scaler"]
                print(f"[SVM] Loaded model from {_MODEL_FILE} "
                      f"({payload['n_genuine']} genuine / "
                      f"{payload['n_forged']} forged samples)")
                return
            else:
                print("[SVM] Training data changed — retraining...")
        except Exception as e:
            print(f"[SVM] Could not load {_MODEL_FILE} ({e}) — retraining...")

    _model, _scaler = _train_and_save()


# ── Build at import time ──────────────────────────────────────────────────────
_load_model()


# ─────────────────────────────────────────────────────────────────────────────
# Public API  (called by app.py)
# ─────────────────────────────────────────────────────────────────────────────

def svm_algo():
    """
    Classify the most-recently-modified image in static/LineSweep_Results.

    Returns:
        "Genuine"  — authenticity probability >= 0.55
        "Forged"   — authenticity probability <  0.55
    """
    global _model, _scaler

    if _model is None or _scaler is None:
        _load_model()

    image_test_paths = "static/LineSweep_Results"
    all_files = [f for f in listdir(image_test_paths)
                 if not f.startswith('.')]

    if not all_files:
        print("[SVM] No test images found.")
        return "No test images"

    # Always use the most recently modified file
    all_files_full = [os.path.join(image_test_paths, f) for f in all_files]
    latest = max(all_files_full, key=os.path.getmtime)
    print(f"[SVM] Processing: {os.path.basename(latest)}")

    try:
        feat    = np.array(_get_features(latest), dtype=np.float32).reshape(1, -1)
        feat_sc = _scaler.transform(feat)
        proba   = _model.predict_proba(feat_sc)[0]
        genuine_prob = float(proba[1])   # index 1 = genuine (label=1)

        print(f"[SVM] Authenticity score: {genuine_prob:.1%}")

        result = "Genuine" if genuine_prob >= 0.55 else "Forged"
        print(f"[SVM] Result: {result}")
        return result

    except Exception as e:
        print(f"[SVM] Error processing image: {e}")
        return "Forged"


# ─────────────────────────────────────────────────────────────────────────────
# Optional: force a retrain (call from shell or admin route)
# ─────────────────────────────────────────────────────────────────────────────

def force_retrain():
    """Delete the saved model and rebuild from scratch."""
    global _model, _scaler
    if os.path.exists(_MODEL_FILE):
        os.remove(_MODEL_FILE)
        print(f"[SVM] Deleted {_MODEL_FILE}")
    _model, _scaler = _train_and_save()