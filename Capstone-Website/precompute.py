"""
Run this ONCE before starting the Flask app:
    python precompute.py

It processes all 290 training images, computes SIFT + contour features,
builds the k-means vocabulary, and saves everything to training_cache.pkl.
The cache is then loaded at startup in svm.py — no reprocessing on each request.
"""

import numpy as np
from os import listdir
import cv2
from PIL import Image
import imagehash
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import joblib
import preproc
import features
import time

start = time.time()

GENUINE_PATH = "data/genuine"
FORGED_PATH  = "data/forged"
K_CLUSTERS   = 500
N_USERS      = 29

print("=== eSig Training Cache Builder ===")
print(f"Loading filenames...")

genuine_filenames = sorted(listdir(GENUINE_PATH))
forged_filenames  = sorted(listdir(FORGED_PATH))

print(f"  Genuine: {len(genuine_filenames)} images")
print(f"  Forged:  {len(forged_filenames)} images")

# Group by user ID
genuine_groups = [[] for _ in range(N_USERS)]
forged_groups  = [[] for _ in range(N_USERS)]

for name in genuine_filenames:
    uid = int(name.split('_')[0][-3:])
    genuine_groups[uid - 1].append(name)

for name in forged_filenames:
    uid = int(name.split('_')[0][-3:])
    forged_groups[uid - 1].append(name)

# Single SIFT detector instance (not recreated per image)
try:
    sift_detector = cv2.xfeatures2d.SIFT_create()
except AttributeError:
    sift_detector = cv2.SIFT_create()

def extract_features(image_path):
    """Returns (sift_descriptors, contour_feature_vector_12d)"""
    preprocessed = preproc.preproc(image_path, display=False)
    
    kp, des = sift_detector.detectAndCompute(preprocessed, None)
    
    aspect_ratio, bounding_rect_area, convex_hull_area, contours_area = \
        features.get_contour_features(preprocessed.copy(), display=False)
    
    ratio     = features.Ratio(preprocessed.copy())
    c0, c1    = features.Centroid(preprocessed.copy())
    ecc, sol  = features.EccentricitySolidity(preprocessed.copy())
    (sk0, sk1), (ku0, ku1) = features.SkewKurtosis(preprocessed.copy())

    contour_vec = [
        aspect_ratio,
        convex_hull_area / bounding_rect_area,
        contours_area / bounding_rect_area,
        ratio, c0, c1, ecc, sol, sk0, sk1, ku0, ku1
    ]
    return des, contour_vec

# Per-user cache: store (sift_des, contour_vec, label) for each image
# label: 1 = forged, 2 = genuine
user_caches = []

total_images = len(genuine_filenames) + len(forged_filenames)
processed = 0

for i in range(N_USERS):
    print(f"\nUser {i+1:02d}/{N_USERS}  ({len(genuine_groups[i])} genuine, {len(forged_groups[i])} forged)")
    user_data = []  # list of (des, contour_vec, label)

    for name in genuine_groups[i]:
        path = f"{GENUINE_PATH}/{name}"
        try:
            des, cvec = extract_features(path)
            user_data.append((des, cvec, 2))  # 2 = genuine
        except Exception as e:
            print(f"  SKIP {name}: {e}")
        processed += 1
        if processed % 20 == 0:
            elapsed = time.time() - start
            print(f"  [{processed}/{total_images}] {elapsed:.0f}s elapsed")

    for name in forged_groups[i]:
        path = f"{FORGED_PATH}/{name}"
        try:
            des, cvec = extract_features(path)
            user_data.append((des, cvec, 1))  # 1 = forged
        except Exception as e:
            print(f"  SKIP {name}: {e}")
        processed += 1

    user_caches.append(user_data)

print(f"\nBuilding per-user k-means vocabularies (k={K_CLUSTERS})...")
print("This is the slow part — ~30s per user group...")

user_models = []  # list of (voc, stdSlr, clf) per user

for i, user_data in enumerate(user_caches):
    if len(user_data) < 10:
        print(f"  User {i+1}: skipping (only {len(user_data)} samples)")
        user_models.append(None)
        continue

    # Stack all SIFT descriptors for this user
    all_des = [d for d, c, l in user_data if d is not None]
    if not all_des:
        user_models.append(None)
        continue

    descriptors = np.vstack(all_des)
    effective_k = min(K_CLUSTERS, descriptors.shape[0])
    voc, _ = kmeans(descriptors, effective_k, 1)

    # Build feature matrix
    n = len(user_data)
    feat_dim = effective_k + 12
    X = np.zeros((n, feat_dim), "float32")
    y = np.zeros(n, dtype=int)

    for ii, (des, cvec, label) in enumerate(user_data):
        if des is not None:
            words, _ = vq(des, voc)
            for w in words:
                if w < effective_k:
                    X[ii][w] += 1
        for j in range(12):
            X[ii][effective_k + j] = cvec[j]
        y[ii] = label

    # Fit scaler on training split ONLY — test image uses this same scaler at inference
    n_genuine = len(genuine_groups[i])
    train_gen_X = X[0:3]
    train_for_X = X[n_genuine:n_genuine+3]
    train_X = np.concatenate((train_for_X, train_gen_X))
    train_y = np.array([1]*len(train_for_X) + [2]*len(train_gen_X))

    train_scaler = StandardScaler().fit(train_X)
    train_X_scaled = train_scaler.transform(train_X)

    clf = LinearSVC()
    clf.fit(train_X_scaled, train_y)

    user_models.append({
        "voc":          voc,
        "k":            effective_k,
        "train_scaler": train_scaler,
        "clf":          clf,
    })
    print(f"  User {i+1:02d}: done (vocab k={effective_k})")

# Save everything to disk
cache = {
    "user_models":    user_models,
    "n_users":        N_USERS,
}

joblib.dump(cache, "training_cache.pkl")

elapsed = time.time() - start
print(f"\n=== Done! training_cache.pkl saved ({elapsed:.0f}s total) ===")
print("You can now start the Flask app — inference will be fast.")