"""
02_calibrate.py
===============
Computes the camera's intrinsic matrix and distortion coefficients
from the detected checkerboard corners.

OUTPUT:
  - calibration_results.npz   — saved K, dist, rvecs, tvecs for later scripts
  - output/reprojection/      — per-image reprojection error visualisations

KEY CONCEPT:
  Calibration solves the following: given many 2D image points whose 3D
  positions we know (the checkerboard corners), find the intrinsic matrix K
  and distortion coefficients that map the 3D points to the 2D points with
  minimum error.

  The 'reprojection error' measures how well the solved K + dist project the
  known 3D points back onto the original 2D detections. Under 1.0 pixel
  is a good calibration. Under 0.5 is excellent.
"""

import cv2
import numpy as np
import glob
import os

# ── CONFIG ────────────────────────────────────────────────────────────────────
IMAGES_DIR = "images"
GRID = (6, 9)              # inner corners (rows, cols) — same as Step 1
SQUARE_SIZE = 1.0          # arbitrary unit. Use real mm if you want metric K.
                           # Doesn't affect K shape, only physical interpretation.
OUTPUT_DIR = "output/reprojection"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── BUILD 3D OBJECT POINTS ────────────────────────────────────────────────────
# We assume the checkerboard lies on the Z=0 plane in its own coordinate frame.
# objp is the same for every image — what changes is the camera's pose.
#
# Shape: (54, 3) for a 6x9 grid. Each row is (x, y, 0).
objp = np.zeros((GRID[0] * GRID[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:GRID[1], 0:GRID[0]].T.reshape(-1, 2) * SQUARE_SIZE

# Storage for matching pairs across all images
objpoints = []   # list of 3D points (one entry per image, all identical = objp)
imgpoints = []   # list of 2D points (detected corners per image)
img_shape = None
used_paths = []

# ── DETECT CORNERS IN ALL IMAGES ──────────────────────────────────────────────
image_paths = sorted(glob.glob(os.path.join(IMAGES_DIR, "*.jpg")))
print(f"Processing {len(image_paths)} images...\n")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

for path in image_paths:
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if img_shape is None:
        img_shape = gray.shape[::-1]   # (width, height) — order matters for cv2

    ret, corners = cv2.findChessboardCorners(
        gray, GRID,
        cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    if ret:
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_refined)
        used_paths.append(path)

print(f"Using {len(objpoints)} images for calibration\n")

# ── CALIBRATE ─────────────────────────────────────────────────────────────────
# cv2.calibrateCamera returns:
#   K       — 3x3 intrinsic matrix
#   dist    — distortion coefficients (k1, k2, p1, p2, k3)
#   rvecs   — rotation vector per image (Rodrigues form)
#   tvecs   — translation vector per image
# rvecs and tvecs are the *extrinsics* — where the camera was for each shot.
print("Running cv2.calibrateCamera...")
ret_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, img_shape, None, None
)

print(f"\nOverall RMS error from cv2.calibrateCamera: {ret_rms:.4f} pixels")

# ── PRINT INTRINSIC MATRIX WITH PHYSICAL MEANING ──────────────────────────────
print("\n" + "=" * 60)
print("INTRINSIC MATRIX K")
print("=" * 60)
print(K)
print(f"\n  fx = {K[0,0]:.2f} px   focal length in pixels (x-axis)")
print(f"  fy = {K[1,1]:.2f} px   focal length in pixels (y-axis)")
print(f"  cx = {K[0,2]:.2f} px   principal point x (optical axis hits sensor)")
print(f"  cy = {K[1,2]:.2f} px   principal point y")
print(f"\n  Sensor size: {img_shape[0]} x {img_shape[1]} px")
print(f"  Image centre: ({img_shape[0]/2:.1f}, {img_shape[1]/2:.1f})")
print(f"  Principal point offset from centre: "
      f"({K[0,2]-img_shape[0]/2:+.2f}, {K[1,2]-img_shape[1]/2:+.2f}) px")

# ── PRINT DISTORTION COEFFICIENTS ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("DISTORTION COEFFICIENTS")
print("=" * 60)
k1, k2, p1, p2, k3 = dist.ravel()
print(f"  k1 = {k1:+.5f}   radial")
print(f"  k2 = {k2:+.5f}   radial")
print(f"  k3 = {k3:+.5f}   radial")
print(f"  p1 = {p1:+.5f}   tangential")
print(f"  p2 = {p2:+.5f}   tangential")
print(f"\n  Sign of k1: {'barrel distortion' if k1 < 0 else 'pincushion distortion'}")

# ── PER-IMAGE REPROJECTION ERROR ──────────────────────────────────────────────
# For each image, project the known 3D points back onto the image using the
# solved K, dist, and that image's pose (rvec, tvec). Compare to the actual
# detected corners. Distance = error.
print("\n" + "=" * 60)
print("PER-IMAGE REPROJECTION ERROR")
print("=" * 60)

per_image_errors = []
for i, (objp_i, imgp_i) in enumerate(zip(objpoints, imgpoints)):
    projected, _ = cv2.projectPoints(objp_i, rvecs[i], tvecs[i], K, dist)
    error = cv2.norm(imgp_i, projected, cv2.NORM_L2) / len(projected)
    per_image_errors.append(error)
    name = os.path.basename(used_paths[i])
    print(f"  {name:<20} {error:.4f} px")

mean_err = np.mean(per_image_errors)
print(f"\n  MEAN reprojection error: {mean_err:.4f} pixels")

if mean_err < 0.5:
    print("  ✓ Excellent calibration")
elif mean_err < 1.0:
    print("  ✓ Good calibration")
else:
    print("  ⚠ Poor calibration — consider removing high-error images")

# ── SAVE RESULTS ──────────────────────────────────────────────────────────────
np.savez(
    "calibration_results.npz",
    K=K, dist=dist, rvecs=rvecs, tvecs=tvecs,
    img_shape=img_shape, mean_error=mean_err
)
print(f"\n  Saved → calibration_results.npz")
print(f"  Used by: 03_undistort.py, 04_homography.py")
