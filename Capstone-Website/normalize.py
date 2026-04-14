"""
normalize.py - Cleans cheque signature crops to match training format.
Uses morphological closing to merge strokes then removes noise.
Result: ink% in 4-10% range matching CEDAR training data.
"""

import cv2
import numpy as np
import os


def normalize_signature(image_path, output_path=None):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Step 1: Otsu threshold to find ink
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Step 2: Close gaps in strokes — merges nearby ink pixels into solid strokes
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)

    # Step 3: Keep only large connected components (real strokes, not watermark dots)
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(closed)
    cleaned = np.zeros_like(closed)
    for label in range(1, n_labels):
        if stats[label, cv2.CC_STAT_AREA] >= 500:
            cleaned[labels == label] = 255

    # Step 4: Erode back to remove closing expansion
    cleaned = cv2.erode(cleaned, np.ones((3, 3), np.uint8))

    # Step 5: Tight crop to signature bounds
    coords = cv2.findNonZero(cleaned)
    if coords is None:
        print(f"  Warning: no ink found in {os.path.basename(image_path)}, using original")
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        if output_path:
            cv2.imwrite(output_path, result)
        return result

    x, y, w, h = cv2.boundingRect(coords)
    pad = 10
    x = max(0, x - pad);     y = max(0, y - pad)
    w = min(cleaned.shape[1] - x, w + 2*pad)
    h = min(cleaned.shape[0] - y, h + 2*pad)
    cropped = cleaned[y:y+h, x:x+w]

    # Step 6: Resize to 200x90, keep aspect ratio
    target_w, target_h = 200, 90
    ch, cw = cropped.shape
    scale  = min(target_w / cw, target_h / ch)
    new_w  = max(1, int(cw * scale))
    new_h  = max(1, int(ch * scale))
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Step 7: White background, dark ink — matches CEDAR training format
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    y_off  = (target_h - new_h) // 2
    x_off  = (target_w - new_w) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w][resized > 127] = 0

    result = cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)

    if output_path:
        ink_pct = np.sum(canvas < 128) / canvas.size * 100
        print(f"  Normalized: {os.path.basename(image_path)} ink={ink_pct:.1f}% size={new_w}x{new_h}")
        cv2.imwrite(output_path, result)

    return result


def normalize_all(folder="static/LineSweep_Results"):
    files = [f for f in os.listdir(folder) if not f.startswith('.')]
    if not files:
        print("No files to normalize.")
        return
    for fname in files:
        fpath = os.path.join(folder, fname)
        try:
            normalize_signature(fpath, output_path=fpath)
        except Exception as e:
            print(f"  Normalize failed for {fname}: {e}")
    print(f"Normalization done for {len(files)} file(s).")