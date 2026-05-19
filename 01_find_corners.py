"""
01_find_corners.py
==================
Detects checkerboard corners in calibration images.

The checkerboard pattern gives us 'control points' — points whose
real-world 3D positions we know exactly (they're on a flat plane
with known spacing). By finding the same points in multiple images
taken from different angles, we can solve for the camera's intrinsic
parameters.

Grid size (6, 9): number of INNER corners — not squares.
A 7x10 square checkerboard has 6x9 inner corners.
"""

import cv2
import numpy as np
import glob
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGES_DIR = "images"
GRID = (6, 9)          # (rows, cols) of INNER corners — adjust if needed
OUTPUT_DIR = "output/corners"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── FIND CORNERS ──────────────────────────────────────────────────────────────
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
print(f"Found {len(image_paths)} images\n")

found = []
failed = []

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    # CALIB_CB_ADAPTIVE_THRESH handles uneven lighting
    # CALIB_CB_NORMALIZE_IMAGE normalises before thresholding
    ret, corners = cv2.findChessboardCorners(
        gray, GRID,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        # Refine corner positions to sub-pixel accuracy
        # This significantly improves calibration quality
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # Draw and save
        img_corners = img.copy()
        cv2.drawChessboardCorners(img_corners, GRID, corners_refined, ret)

        name = os.path.basename(path).replace(".jpg", "_corners.jpg")
        cv2.imwrite(os.path.join(OUTPUT_DIR, name), img_corners)
        found.append(path)
        print(f"  ✓ {os.path.basename(path)} — corners found")
    else:
        failed.append(path)
        print(f"  ✗ {os.path.basename(path)} — corners NOT found")

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\nResult: {len(found)}/{len(image_paths)} images usable")
if failed:
    print(f"Failed: {[os.path.basename(p) for p in failed]}")
    print("Tip: check GRID size matches your actual checkerboard inner corners")

print(f"\nCorner visualisations saved to: {OUTPUT_DIR}/")