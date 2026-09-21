import argparse
import cv2
import numpy as np


def load_calibration(path):
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"], tuple(data["image_size"])


def scale_camera_matrix(camera_matrix, calib_image_size, actual_image_size):
    calib_w, calib_h = calib_image_size
    actual_w, actual_h = actual_image_size

    if (calib_w, calib_h) == (actual_w, actual_h):
        return camera_matrix

    scale_x = actual_w / calib_w
    scale_y = actual_h / calib_h

    print(f"\n[warning] object photo resolution ({actual_w}x{actual_h}) does not match "
          f"calibration resolution ({calib_w}x{calib_h}).")
    print(f"          Scaling camera matrix by ({scale_x:.4f}, {scale_y:.4f}) to compensate.")
    print("          For best accuracy, shoot calibration and measurement photos at the same resolution/zoom.\n")

    scaled = camera_matrix.copy()
    scaled[0, 0] *= scale_x  # fx
    scaled[0, 2] *= scale_x  # cx
    scaled[1, 1] *= scale_y  # fy
    scaled[1, 2] *= scale_y  # cy
    return scaled


def undistort_points(points_px, camera_matrix, dist_coeffs):
    pts = np.array(points_px, dtype=np.float32).reshape(-1, 1, 2)
    undistorted = cv2.undistortPoints(pts, camera_matrix, dist_coeffs, P=camera_matrix)
    return undistorted.reshape(-1, 2)


def pixel_length_to_real(p1, p2, distance_mm, camera_matrix):
    """Core formula: L = (pixel_distance * Z) / f"""
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]

    dx_px = p2[0] - p1[0]
    dy_px = p2[1] - p1[1]

    dx_mm = (dx_px * distance_mm) / fx
    dy_mm = (dy_px * distance_mm) / fy

    real_length_mm = float(np.sqrt(dx_mm ** 2 + dy_mm ** 2))
    return real_length_mm, dx_mm, dy_mm



_clicked_points = []


def _on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(_clicked_points) < 2:
        _clicked_points.append((x, y))


def pick_two_points(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(image_path)

    display = img.copy()
    cv2.namedWindow("click two points, then press any key")
    cv2.setMouseCallback("click two points, then press any key", _on_mouse)

    while True:
        vis = display.copy()
        for pt in _clicked_points:
            cv2.circle(vis, pt, 6, (0, 0, 255), -1)
        if len(_clicked_points) == 2:
            cv2.line(vis, _clicked_points[0], _clicked_points[1], (0, 255, 0), 2)
        cv2.imshow("click two points, then press any key", vis)
        key = cv2.waitKey(20)
        if len(_clicked_points) == 2 and key != -1:
            break
        if key == 27:  # esc
            break

    cv2.destroyAllWindows()
    return _clicked_points


def main():
    parser = argparse.ArgumentParser(description="Measure a real-world length from an image")
    parser.add_argument("--image", required=True, help="path to the object photo")
    parser.add_argument("--distance", type=float, required=True, help="camera-to-object distance in mm")
    parser.add_argument("--calib", default="camera_calib.npz", help="calibration file from calibration.py")
    parser.add_argument("--p1", nargs=2, type=float, default=None, help="pixel coords x y of point 1")
    parser.add_argument("--p2", nargs=2, type=float, default=None, help="pixel coords x y of point 2")
    args = parser.parse_args()

    camera_matrix, dist_coeffs, calib_image_size = load_calibration(args.calib)

    img_check = cv2.imread(args.image)
    if img_check is None:
        raise FileNotFoundError(args.image)
    actual_image_size = (img_check.shape[1], img_check.shape[0])
    camera_matrix = scale_camera_matrix(camera_matrix, calib_image_size, actual_image_size)

    if args.p1 and args.p2:
        p1_raw, p2_raw = tuple(args.p1), tuple(args.p2)
    else:
        print("Click the two endpoints of the edge you want to measure, then press any key.")
        pts = pick_two_points(args.image)
        if len(pts) < 2:
            print("Need two points - aborting.")
            return
        p1_raw, p2_raw = pts[0], pts[1]

    p1_u, p2_u = undistort_points([p1_raw, p2_raw], camera_matrix, dist_coeffs)

    pixel_dist = float(np.hypot(p2_u[0] - p1_u[0], p2_u[1] - p1_u[1]))
    real_mm, dx_mm, dy_mm = pixel_length_to_real(p1_u, p2_u, args.distance, camera_matrix)

    print("\n===== Measurement result =====")
    print(f"Clicked points (raw px):     {p1_raw}, {p2_raw}")
    print(f"Undistorted points (px):     {tuple(p1_u)}, {tuple(p2_u)}")
    print(f"Pixel distance:              {pixel_dist:.2f} px")
    print(f"Distance to object (Z):      {args.distance:.1f} mm")
    print(f"Real-world length:           {real_mm:.2f} mm  ({real_mm/10:.2f} cm)")


if __name__ == "__main__":
    main()