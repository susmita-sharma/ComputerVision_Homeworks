#10*7

import argparse
import glob
import os
import sys

import cv2
import numpy as np

def find_corners(image_paths, pattern_size, square_size_mm, show=False):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size_mm

    object_points = []   
    image_points = []    
    image_size = None
    used, skipped = 0, 0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for path in image_paths:
        img = cv2.imread(path)
        if img is None:
            print(f"  [skip] could not read {path}")
            skipped += 1
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])  # (width, height)

        found, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print(f"  [skip] no chessboard found in {os.path.basename(path)}")
            skipped += 1
            continue

        corners_refined = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )

        object_points.append(objp)
        image_points.append(corners_refined)
        used += 1

        if show:
            vis = img.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners_refined, found)
            cv2.imshow("corners", vis)
            cv2.waitKey(300)

    if show:
        cv2.destroyAllWindows()

    print(f"\nUsed {used} images, skipped {skipped} (no board detected / unreadable).")
    return object_points, image_points, image_size


def calibrate(object_points, image_points, image_size):
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    total_error = 0
    per_image_error = []
    for i in range(len(object_points)):
        projected, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
        )
        observed = np.asarray(image_points[i], dtype=np.float64).reshape(-1, 2)
        predicted = np.asarray(projected, dtype=np.float64).reshape(-1, 2)
        error = float(np.sqrt(np.sum((observed - predicted) ** 2, axis=1)).mean())
        per_image_error.append(error)
        total_error += error

    mean_error = total_error / len(object_points)

    return camera_matrix, dist_coeffs, mean_error, per_image_error


def main():
    parser = argparse.ArgumentParser(description="OpenCV chessboard camera calibration")
    parser.add_argument("--images", default="calibration_images", help="folder of calibration photos")
    parser.add_argument("--cols", type=int, default=9, help="inner corners across the board width")
    parser.add_argument("--rows", type=int, default=6, help="inner corners across the board height")
    parser.add_argument("--square", type=float, default=25.0, help="chessboard square size in mm")
    parser.add_argument("--out", default="camera_calib.npz", help="output file for calibration results")
    parser.add_argument("--show", action="store_true", help="display detected corners while running")
    args = parser.parse_args()

    image_paths = sorted(
        glob.glob(os.path.join(args.images, "*.jpg"))
        + glob.glob(os.path.join(args.images, "*.jpeg"))
        + glob.glob(os.path.join(args.images, "*.png"))
    )

    if not image_paths:
        print(f"No images found in '{args.images}'. Drop your chessboard photos there and try again.")
        sys.exit(1)

    print(f"Found {len(image_paths)} candidate images in '{args.images}'.")

    object_points, image_points, image_size = find_corners(
        image_paths, (args.cols, args.rows), args.square, show=args.show
    )

    if len(object_points) < 10:
        print(
            f"\nOnly {len(object_points)} usable images - calibration needs at least "
            "10-15 good ones to be reliable. Take a few more photos and re-run."
        )
        if len(object_points) < 3:
            sys.exit(1)

    camera_matrix, dist_coeffs, mean_error, per_image_error = calibrate(
        object_points, image_points, image_size
    )

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]

    print("\n===== Calibration result =====")
    print(f"Image size (w x h):     {image_size}")
    print(f"fx, fy (pixels):        {fx:.2f}, {fy:.2f}")
    print(f"cx, cy (principal pt):  {cx:.2f}, {cy:.2f}")
    print(f"Distortion coeffs:      {dist_coeffs.ravel()}")
    print(f"Mean reprojection err:  {mean_error:.4f} px")
    worst = int(np.argmax(per_image_error))
    print(f"Worst image:            #{worst} ({per_image_error[worst]:.4f} px) - "
          f"drop it and re-run if this looks off")

    np.savez(
        args.out,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=mean_error,
        image_size=np.array(image_size),
    )
    print(f"\nSaved calibration to '{args.out}'.")


if __name__ == "__main__":
    main()