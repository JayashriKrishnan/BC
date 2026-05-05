"""
signet.py - Wrapper for the SigNet Siamese network model.

Model: signet_model.keras  (Keras 3 Functional, Siamese architecture)
Input: two images, each (155, 220, 3)  float32
Output: Euclidean distance between the two 128-dim embeddings
        distance < THRESHOLD  →  "Match"   (same signer)
        distance >= THRESHOLD →  "No Match" (different signer / forgery)

Preprocessing (must match training exactly):
    1. cv2.imread()        — read as BGR, 3-channel uint8
    2. cv2.resize(220,155) — width=220, height=155  →  shape (155,220,3)
    3. cv2.bitwise_not()   — invert pixels (dark ink on white → white on black)
    4. / 255.0             — normalize to [0.0, 1.0]

NOTE — why we rebuild the architecture in code:
    The model was saved with Keras 3.13.2 which requires Python ≥ 3.11.
    This project runs on Python 3.10, so the highest available Keras is 3.12.1.
    Keras 3.12.1 cannot deserialize the saved config (it doesn't know the
    'quantization_config' field added in 3.13.x).
    Solution: rebuild the exact same architecture here and load only the weights
    from the HDF5 file inside the .keras zip — no config deserialization needed.
"""

import os
import zipfile
import tempfile
import numpy as np
import cv2

# ── Configurable threshold ─────────────────────────────────────────────────────
# Theoretically grounded at 0.5 (contrastive loss margin=1, midpoint=0.5).
# Lower → stricter (fewer false-genuine).  Higher → more lenient.
SIGNET_THRESHOLD = 0.5

# Path to the model file, relative to this module's directory (one level up)
_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "signet_model.keras"
)

# Module-level model cache
_model = None


def _build_siamese_model():
    """
    Rebuild the SigNet Siamese architecture exactly as saved in the .keras config.

    Branch architecture (shared weights):
        Input(155,220,3)
        Conv2D(96,11×11,valid,linear) → BatchNorm → ReLU → MaxPool(3×3,stride=3) → Dropout(0.3)
        Conv2D(384,3×3,valid,linear)  → ReLU
        Conv2D(256,3×3,valid,linear)  → ReLU
        MaxPool(3×3,stride=3) → Dropout(0.3)
        Flatten → Dense(1024,relu) → Dropout(0.5) → Dense(128,relu)

    Outer model: Lambda(euclidean_distance) on the two branch outputs.
    """
    import keras
    from keras import layers, Model
    import keras.ops as ops

    def euclidean_distance(vects):
        x, y = vects
        diff = x - y
        return ops.sqrt(ops.sum(ops.square(diff), axis=1, keepdims=True) + 1e-7)

    # ── Shared CNN branch ─────────────────────────────────────────────────────
    inp = layers.Input(shape=(155, 220, 3))

    x = layers.Conv2D(96, (11, 11), padding='valid', activation='linear')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv2D(384, (3, 3), padding='valid', activation='linear')(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(256, (3, 3), padding='valid', activation='linear')(x)
    x = layers.Activation('relu')(x)

    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3))(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)

    branch = Model(inp, x, name='siamese_branch')

    # ── Siamese wrapper ───────────────────────────────────────────────────────
    input_a = layers.Input(shape=(155, 220, 3))
    input_b = layers.Input(shape=(155, 220, 3))

    emb_a = branch(input_a)
    emb_b = branch(input_b)

    distance = layers.Lambda(euclidean_distance)([emb_a, emb_b])
    model = Model(inputs=[input_a, input_b], outputs=distance, name='signet')

    return model


def load_model():
    """
    Build the architecture and load weights from signet_model.keras.
    Called once at app startup so the first request is not slow.
    """
    global _model
    if _model is not None:
        return _model

    if not os.path.exists(_MODEL_PATH):
        raise FileNotFoundError(
            f"[Signet] Model file not found: {os.path.abspath(_MODEL_PATH)}"
        )

    _model = _build_siamese_model()

    # Extract model.weights.h5 from inside the .keras zip and load it
    with zipfile.ZipFile(_MODEL_PATH, 'r') as zf:
        with tempfile.TemporaryDirectory() as tmpdir:
            zf.extract('model.weights.h5', tmpdir)
            weights_path = os.path.join(tmpdir, 'model.weights.h5')
            _model.load_weights(weights_path)

    print(f"[Signet] Model loaded from {os.path.abspath(_MODEL_PATH)}")
    return _model


def preprocess_for_signet(image_path):
    """
    Apply the exact preprocessing pipeline used during training.

    Steps (in order):
        1. cv2.imread          → BGR, 3-channel, uint8
        2. cv2.resize(220,155) → shape (155, 220, 3)
        3. cv2.bitwise_not     → invert (white bg → black bg, ink becomes bright)
        4. / 255.0             → normalize to [0.0, 1.0] float32
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"[Signet] Cannot read image: {image_path}")

    img = cv2.resize(img, (220, 155))       # (width, height) in OpenCV → shape (155,220,3)
    img = cv2.bitwise_not(img)              # invert: 255 - pixel
    img = img.astype(np.float32) / 255.0   # normalize to [0.0, 1.0]
    return img


def compare_signatures(cheque_sig_path, genuine_sig_path, threshold=None):
    """
    Compare the extracted cheque signature against the enrolled genuine signature.

    Args:
        cheque_sig_path:  path to the normalized crop from the cheque pipeline
        genuine_sig_path: path to the enrolled genuine signature from the DB
        threshold:        override SIGNET_THRESHOLD if provided

    Returns:
        (distance: float, verdict: str)
        verdict is "Match" if distance < threshold, else "No Match"
    """
    if threshold is None:
        threshold = SIGNET_THRESHOLD

    model = load_model()

    img1 = preprocess_for_signet(cheque_sig_path)
    img2 = preprocess_for_signet(genuine_sig_path)

    # Add batch dimension: (155,220,3) → (1,155,220,3)
    batch1 = np.expand_dims(img1, axis=0)
    batch2 = np.expand_dims(img2, axis=0)

    # Model returns shape (1, 1) — Euclidean distance
    raw      = _model.predict([batch1, batch2], verbose=0)
    distance = float(raw.flat[0])

    verdict = "Match" if distance < threshold else "No Match"
    print(f"[Signet] Distance: {distance:.4f}  (threshold={threshold})  → {verdict}")
    return distance, verdict
