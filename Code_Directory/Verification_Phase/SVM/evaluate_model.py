"""
evaluate_model.py  —  E-Signify SVM Evaluation Script
-------------------------------------------------------
Place this file inside:  Code_Directory/Verification_Phase/SVM/
Run:  python evaluate_model.py
"""

import numpy as np
import pickle
import cv2
from PIL import Image
from os import listdir
import imagehash
from scipy.cluster.vq import kmeans, vq
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt
import preproc
import features

GENUINE_DIR = "data/genuine"
FORGED_DIR  = "data/forged"
MODEL_PATH  = "model.pkl"
K           = 500
NUM_USERS   = 29
OUTPUT_IMG  = "confusion_matrix.png"


def get_sift_des(image):
    try:
        s = cv2.xfeatures2d.SIFT_create()
    except AttributeError:
        s = cv2.SIFT_create()
    _, des = s.detectAndCompute(image, None)
    return des


def extract(image_path):
    pre = preproc.preproc(image_path, display=False)
    ar, bra, cha, ca = features.get_contour_features(pre.copy(), display=False)
    ratio = features.Ratio(pre.copy())
    c0, c1 = features.Centroid(pre.copy())
    ecc, sol = features.EccentricitySolidity(pre.copy())
    (sk0, sk1), (ku0, ku1) = features.SkewKurtosis(pre.copy())
    cf = [ar, cha/bra, ca/bra, ratio, c0, c1, ecc, sol, sk0, sk1, ku0, ku1]
    return cf, get_sift_des(pre.copy())


def bow_matrix(des_list, cf_list, voc, k):
    n = len(des_list)
    mat = np.zeros((n, k + 12), "float32")
    for i in range(n):
        if des_list[i] is not None:
            words, _ = vq(des_list[i], voc)
            for w in words:
                mat[i][w] += 1
        for j in range(12):
            mat[i][k + j] = cf_list[i][j]
    return mat


def load_user(uid, g_dir, f_dir, g_files, f_files):
    g_names = [n for n in g_files if int(n.split('_')[0][-3:]) == uid + 1]
    f_names = [n for n in f_files if int(n.split('_')[0][-3:]) == uid + 1]
    des, cf, labs = [], [], []
    for name in g_names:
        try:
            c, d = extract(g_dir + "/" + name)
            cf.append(c); des.append(d); labs.append(2)
        except Exception as e:
            print(f"  skip {name}: {e}")
    for name in f_names:
        try:
            c, d = extract(f_dir + "/" + name)
            cf.append(c); des.append(d); labs.append(1)
        except Exception as e:
            print(f"  skip {name}: {e}")
    return des, cf, labs


def main():
    print(f"Loading model: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("Model loaded.\n")

    g_files = listdir(GENUINE_DIR)
    f_files = listdir(FORGED_DIR)
    print(f"Genuine: {len(g_files)}  |  Forged: {len(f_files)}\n")

    all_preds, all_labs = [], []

    for i in range(NUM_USERS):
        print(f"User group {i+1}/{NUM_USERS} ...", end=" ", flush=True)
        des, cf, labs = load_user(i, GENUINE_DIR, FORGED_DIR, g_files, f_files)
        if len(des) < 2:
            print("skipped (too few samples)")
            continue
        valid = [d for d in des if d is not None]
        if not valid:
            print("skipped (no SIFT)")
            continue
        all_d = np.vstack(valid)
        ek = min(K, all_d.shape[0])
        try:
            voc, _ = kmeans(all_d, ek, 1)
        except Exception as e:
            print(f"kmeans error: {e}")
            continue
        mat = bow_matrix(des, cf, voc, ek)
        mat = StandardScaler().fit_transform(mat)
        try:
            preds = model.predict(mat)
            all_preds.extend(preds.tolist())
            all_labs.extend(labs)
            print(f"done ({len(labs)} samples)")
        except Exception as e:
            print(f"predict error: {e}")

    if not all_preds:
        print("\nNo predictions generated. Check data paths.")
        return

    y_true = np.array(all_labs)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100
    print("\n" + "="*55)
    print(f"  Overall Accuracy : {acc:.2f}%")
    print("="*55)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred,
          target_names=["Forged (1)", "Genuine (2)"], digits=4))

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0
    frr = fn / (fn + tp) * 100 if (fn + tp) > 0 else 0

    print("Confusion Matrix Breakdown:")
    print(f"  True Genuine   (TP): {tp}")
    print(f"  True Forged    (TN): {tn}")
    print(f"  False Accepted (FP): {fp}  <- forged passed as genuine")
    print(f"  False Rejected (FN): {fn}  <- genuine wrongly flagged")
    print(f"\n  FAR : {far:.2f}%")
    print(f"  FRR : {frr:.2f}%")

    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm, display_labels=["Forged", "Genuine"]).plot(
        ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("E-Signify — Confusion Matrix", fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(OUTPUT_IMG, dpi=150)
    print(f"\nSaved: {OUTPUT_IMG}")
    plt.show()


if __name__ == "__main__":
    main()