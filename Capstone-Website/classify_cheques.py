"""
classify_cheques.py

Takes a folder of cheque images, runs the full pipeline on each one,
and COPIES them into genuine/forged output folders.

Original files are never moved or deleted.

Usage:
    python classify_cheques.py \
        --input  C:/path/to/cheque_images \
        --genuine C:/path/to/output/genuine \
        --forged  C:/path/to/output/forged

Example:
    python classify_cheques.py --input D:/cheques --genuine D:/results/genuine --forged D:/results/forged
"""

import os
import shutil
import argparse
import cv2
import numpy as np
from skimage.morphology import skeletonize

# ── Import pipeline modules (must run from Capstone-Website/) ─────────────────
import ocr
import lineSweep
import normalize
import svm


SUPPORTED = ('.jpg', '.jpeg', '.png', '.gif')


def process_single_cheque(image_path):
    """
    Runs the full pipeline on one cheque image.
    Returns "Genuine" or "Forged".
    """
    # Step 1: Copy image into static/uploads (where ocr.py reads from)
    uploads_dir = "static/uploads"
    ls_dir      = "static/LineSweep_Results"
    ocr_dir     = "static/OCR_Results"

    # Clear previous run results
    for folder in [uploads_dir, ocr_dir, ls_dir]:
        for f in os.listdir(folder):
            try:
                os.remove(os.path.join(folder, f))
            except:
                pass

    # Copy cheque into uploads
    filename = os.path.basename(image_path)
    dest     = os.path.join(uploads_dir, filename)
    shutil.copy2(image_path, dest)

    # Step 2: OCR — finds and crops the signature region
    try:
        ocr.ocr_algo()
    except Exception as e:
        print(f"    OCR failed: {e}")
        return "Error"

    # Check OCR produced output
    ocr_files = [f for f in os.listdir(ocr_dir) if not f.startswith('.')]
    if not ocr_files:
        print(f"    OCR found no signature region — skipping")
        return "Error"

    # Step 3: LineSweep — tight crops to just the ink
    try:
        lineSweep.lineSweep_algo()
    except Exception as e:
        print(f"    LineSweep failed: {e}")
        return "Error"

    ls_files = [f for f in os.listdir(ls_dir) if not f.startswith('.')]
    if not ls_files:
        print(f"    LineSweep produced no output — skipping")
        return "Error"

    # Step 4: Normalize — remove watermark, resize to match training format
    try:
        normalize.normalize_all(ls_dir)
    except Exception as e:
        print(f"    Normalize failed: {e}")

    # Step 5: SVM — classify
    try:
        result = svm.svm_algo()
    except Exception as e:
        print(f"    SVM failed: {e}")
        return "Error"

    return result


def classify_folder(input_dir, genuine_dir, forged_dir):
    """
    Processes all cheque images in input_dir.
    Copies each to genuine_dir or forged_dir based on result.
    """
    # Create output folders if they don't exist
    os.makedirs(genuine_dir, exist_ok=True)
    os.makedirs(forged_dir,  exist_ok=True)

    # Get all image files
    all_files = [
        f for f in sorted(os.listdir(input_dir))
        if f.lower().endswith(SUPPORTED)
    ]

    if not all_files:
        print(f"No image files found in: {input_dir}")
        return

    print(f"Found {len(all_files)} cheque image(s) to process")
    print(f"Genuine → {genuine_dir}")
    print(f"Forged  → {forged_dir}")
    print("-" * 50)

    genuine_count = 0
    forged_count  = 0
    error_count   = 0

    for i, fname in enumerate(all_files, 1):
        src_path = os.path.join(input_dir, fname)
        print(f"\n[{i}/{len(all_files)}] {fname}")

        result = process_single_cheque(src_path)

        if result == "Genuine":
            dst = os.path.join(genuine_dir, fname)
            shutil.copy2(src_path, dst)
            print(f"    → GENUINE ✓ (copied to genuine folder)")
            genuine_count += 1

        elif result == "Forged":
            dst = os.path.join(forged_dir, fname)
            shutil.copy2(src_path, dst)
            print(f"    → FORGED  ✗ (copied to forged folder)")
            forged_count += 1

        else:
            print(f"    → ERROR (skipped — could not process)")
            error_count += 1

    # Summary
    print()
    print("=" * 50)
    print(f"DONE — {len(all_files)} cheques processed")
    print(f"  Genuine : {genuine_count}")
    print(f"  Forged  : {forged_count}")
    if error_count:
        print(f"  Errors  : {error_count} (could not extract signature)")
    print(f"\nGenuine folder : {genuine_dir}")
    print(f"Forged  folder : {forged_dir}")
    print(f"Original folder: {input_dir}  ← untouched")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Classify cheque images into genuine/forged folders"
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to folder containing cheque images"
    )
    parser.add_argument(
        "--genuine",
        required=True,
        help="Path to output folder for genuine cheques"
    )
    parser.add_argument(
        "--forged",
        required=True,
        help="Path to output folder for forged cheques"
    )
    args = parser.parse_args()

    # Validate input folder
    if not os.path.isdir(args.input):
        print(f"ERROR: Input folder not found: {args.input}")
        exit(1)

    classify_folder(args.input, args.genuine, args.forged)