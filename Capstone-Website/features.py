"""
features.py — Signature authenticity features.
Focuses on what ACTUALLY separates genuine from forged:
  1. Stroke width variation  (24.6% diff) — pressure variation
  2. Smoothness              ( 8.1% diff) — natural curves vs hesitation
  3. Pen lift count          (29.3% diff) — natural breaks
  4. Retrace ratio           (16.6% diff) — skeleton vs ink
  5. Ink ratio               (27.1% diff) — ink coverage
Plus original geometric features for SVM compatibility.
"""

import numpy as np
import cv2
import os
from scipy import ndimage
from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.morphology import skeletonize


# ── Original features (kept for SVM compatibility) ───────────────────────────

def Ratio(img):
    a = np.sum(img == 255)
    return a / (img.shape[0] * img.shape[1])


def Centroid(img):
    numOfWhites = 0
    a = np.array([0, 0])
    for row in range(len(img)):
        for col in range(len(img[0])):
            if img[row][col] == 255:
                a = np.add(a, np.array([row, col]))
                numOfWhites += 1
    if numOfWhites == 0:
        return 0.5, 0.5
    rowcols = np.array([img.shape[0], img.shape[1]])
    centroid = a / numOfWhites / rowcols
    return centroid[0], centroid[1]


def EccentricitySolidity(img):
    try:
        r = regionprops(img)
        if r:
            return r[0].eccentricity, r[0].solidity
    except Exception:
        pass
    return 0.5, 0.5


def SkewKurtosis(img):
    h, w = img.shape
    x = range(w)
    y = range(h)
    xp = np.sum(img, axis=0)
    yp = np.sum(img, axis=1)
    total = np.sum(img)
    if total == 0:
        return (0, 0), (0, 0)
    cx = np.sum(x * xp) / total
    cy = np.sum(y * yp) / total
    x2 = (x - cx) ** 2
    y2 = (y - cy) ** 2
    sx = np.sqrt(np.sum(x2 * xp) / total) + 1e-10
    sy = np.sqrt(np.sum(y2 * yp) / total) + 1e-10
    x3 = (x - cx) ** 3
    y3 = (y - cy) ** 3
    skewx = np.sum(xp * x3) / (total * sx ** 3)
    skewy = np.sum(yp * y3) / (total * sy ** 3)
    x4 = (x - cx) ** 4
    y4 = (y - cy) ** 4
    kurtx = np.sum(xp * x4) / (total * sx ** 4) - 3
    kurty = np.sum(yp * y4) / (total * sy ** 4) - 3
    return (skewx, skewy), (kurtx, kurty)


def get_contour_features(im, display=False):
    try:
        rect = cv2.minAreaRect(cv2.findNonZero(im))
        box = cv2.boxPoints(rect)
        box = np.int8(box)
        w = np.linalg.norm(box[0] - box[1])
        h = np.linalg.norm(box[1] - box[2])
        if min(w, h) == 0:
            return 1, 1, 1, 1
        aspect_ratio = max(w, h) / min(w, h)
        bounding_rect_area = w * h
        hull = cv2.convexHull(cv2.findNonZero(im))
        contours, _ = cv2.findContours(im.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_area = sum(cv2.contourArea(cnt) for cnt in contours)
        hull_area = cv2.contourArea(hull)
        return aspect_ratio, bounding_rect_area, hull_area, contour_area
    except Exception:
        return 1, 1, 1, 1


# ── NEW: Authenticity-specific features ──────────────────────────────────────

def StrokeWidthVariation(binary_img):
    """
    Measures natural pressure variation in the stroke.
    Genuine: higher variation (natural writing pressure changes)
    Forged:  lower variation (slow careful copying = uniform pressure)
    """
    dist = cv2.distanceTransform(binary_img, cv2.DIST_L2, 5)
    stroke_widths = dist[binary_img > 0]
    if len(stroke_widths) == 0 or stroke_widths.mean() == 0:
        return 0.0
    return float(stroke_widths.std() / stroke_widths.mean())


def StrokeSmoothness(binary_img):
    """
    Measures how smoothly the pen moved.
    Genuine: smooth flowing curves
    Forged:  hesitant jerky strokes
    Returns mean curvature change — lower = smoother
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    scores = []
    for cnt in contours:
        if len(cnt) > 10:
            pts = cnt.reshape(-1, 2).astype(float)
            diffs = np.diff(pts, axis=0)
            angles = np.arctan2(diffs[:, 1], diffs[:, 0])
            changes = np.abs(np.diff(angles))
            changes = np.minimum(changes, 2 * np.pi - changes)
            scores.append(float(np.mean(changes)))
    return float(np.mean(scores)) if scores else 0.5


def PenLiftCount(binary_img):
    """
    Number of connected components = number of times pen was lifted.
    Genuine: natural pen lifts (varies by person)
    Forged:  often fewer (trying to copy in one go) or more (hesitation)
    """
    contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)


def RetraceRatio(binary_img):
    """
    Ratio of skeleton length to total ink area.
    Low ratio = thick strokes with retracing (forged — drawn slowly)
    High ratio = thin natural strokes
    """
    binary_bool = binary_img > 0
    skeleton = skeletonize(binary_bool)
    skeleton_len = float(np.sum(skeleton))
    ink_area = float(np.sum(binary_bool))
    if ink_area == 0:
        return 0.0
    return skeleton_len / ink_area


def get_authenticity_feature_vector(img):
    """
    Returns a 5-element vector of authenticity features.
    These are appended to the standard 12 features for SVM.
    """
    # img should be preprocessed (binary, white ink on black or vice versa)
    # Ensure binary: white ink = 255
    if img.mean() > 127:
        # white background black ink — invert
        binary = cv2.bitwise_not(img)
    else:
        binary = img.copy()

    # Make sure it's proper uint8 binary
    _, binary = cv2.threshold(binary, 127, 255, cv2.THRESH_BINARY)

    sw_cv     = StrokeWidthVariation(binary)
    smooth    = StrokeSmoothness(binary)
    pen_lifts = PenLiftCount(binary)
    retrace   = RetraceRatio(binary)
    ink       = float(np.sum(binary > 0)) / binary.size

    return [sw_cv, smooth, pen_lifts, retrace, ink]