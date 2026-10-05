# HW2 - camera calibration and perspective measurement.
#
# My original module 2 assignment (chessboard calibration -> pixel to real
# world measurement -> accuracy check), just moved into its own folder now
# so it sits next to HW3 and HW4 in the same app. Logic is untouched.
import os
import shutil
import time

import cv2
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash

from Homework import samples
from . import calibration
from . import dimensions
from . import validate as vm

hw2 = Blueprint("hw2", __name__, url_prefix="/hw2", template_folder="templates", static_folder="static")

MODULE_ROOT = os.path.dirname(os.path.abspath(__file__))
UPLOAD_CALIB_DIR = os.path.join(MODULE_ROOT, "static", "calib_uploads")
UPLOAD_OBJ_DIR = os.path.join(MODULE_ROOT, "static", "object_uploads")
CALIB_FILE = os.path.join(MODULE_ROOT, "static", "calib", "camera_calib.npz")
RESULTS_DIR = os.path.join(MODULE_ROOT, "static", "results")

for d in [UPLOAD_CALIB_DIR, UPLOAD_OBJ_DIR, os.path.dirname(CALIB_FILE), RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)


def _corner_preview(paths, pattern_size, out_path, max_dim=1000):
    """Detected chessboard corners drawn on the first photo where the board
    is found (downscaled first so the search is quick) - the visual proof
    for the calibration result."""
    for path in paths:
        img = cv2.imread(path)
        if img is None:
            continue
        s = min(1.0, max_dim / max(img.shape[:2]))
        small = cv2.resize(img, None, fx=s, fy=s) if s < 1 else img
        found, corners = cv2.findChessboardCorners(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), pattern_size)
        if found:
            cv2.drawChessboardCorners(small, pattern_size, corners, found)
            cv2.imwrite(out_path, small, [cv2.IMWRITE_JPEG_QUALITY, 88])
            return True
    return False


def get_current_calibration():
    if not os.path.exists(CALIB_FILE):
        return None
    data = np.load(CALIB_FILE)
    return {
        "camera_matrix": data["camera_matrix"],
        "dist_coeffs": data["dist_coeffs"],
        "reprojection_error": float(data["reprojection_error"]),
        "image_size": tuple(data["image_size"]),
    }


@hw2.route("/calibrate", methods=["GET", "POST"])
def calibrate_view():
    if request.method == "GET":
        sample = samples.load("hw2", "calibration")
        return render_template("hw2/calibration_step.html", result=sample and sample["result"],
                               sample_info=sample and sample["_sample"], calib=get_current_calibration())

    files = request.files.getlist("images")
    cols = int(request.form.get("cols", 9))
    rows = int(request.form.get("rows", 6))
    square = float(request.form.get("square", 25.0))

    if len(files) < 3:
        flash("Please upload chessboard photos (10-15+ recommended).")
        return redirect(url_for("hw2.calibrate_view"))

    # clear old uploads then save the new batch
    for f in os.listdir(UPLOAD_CALIB_DIR):
        if f != ".gitkeep":
            os.remove(os.path.join(UPLOAD_CALIB_DIR, f))

    saved_paths = []
    for f in files:
        path = os.path.join(UPLOAD_CALIB_DIR, f.filename)
        f.save(path)
        saved_paths.append(path)

    object_points, image_points, image_size = calibration.find_corners(
        saved_paths, (cols, rows), square
    )

    if len(object_points) < 3:
        flash("Chessboard wasn't detected in enough images - check the cols/rows values match your "
              "printed board (inner corners, not squares) and try again.")
        return redirect(url_for("hw2.calibrate_view"))

    camera_matrix, dist_coeffs, mean_error, per_image_error = calibration.calibrate(
        object_points, image_points, image_size
    )

    np.savez(
        CALIB_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        reprojection_error=mean_error,
        image_size=np.array(image_size),
    )

    result = {
        "n_used": len(object_points),
        "n_uploaded": len(files),
        "image_size": image_size,
        "fx": float(camera_matrix[0, 0]),
        "fy": float(camera_matrix[1, 1]),
        "cx": float(camera_matrix[0, 2]),
        "cy": float(camera_matrix[1, 2]),
        "dist_coeffs": [round(float(c), 5) for c in dist_coeffs.ravel()],
        "mean_error": mean_error,
    }
    preview = f"{int(time.time() * 1000)}_calibration_corners.jpg"
    if _corner_preview(saved_paths, (cols, rows), os.path.join(RESULTS_DIR, preview)):
        result["corners_url"] = url_for("hw2.static", filename=f"results/{preview}")
    offer = samples.offer("hw2", "calibration", {"result": result})
    return render_template("hw2/calibration_step.html", result=result, calib=get_current_calibration(),
                           sample_offer=offer)


@hw2.route("/measure", methods=["GET", "POST"])
def measure_view():
    calib = get_current_calibration()

    if request.method == "GET":
        sample = samples.load("hw2", "measurement")
        return render_template("hw2/measurement_step.html", calib=calib, image_url=None,
                               result=sample and sample["result"], sample_info=sample and sample["_sample"])

    if calib is None:
        flash("Run calibration before measuring an object.")
        return redirect(url_for("hw2.calibrate_view"))

    # first submit: just uploading the photo, so we can show it for clicking
    if "object_image" in request.files and request.files["object_image"].filename:
        f = request.files["object_image"]
        path = os.path.join(UPLOAD_OBJ_DIR, f.filename)
        f.save(path)
        image_url = url_for("hw2.static", filename=f"object_uploads/{f.filename}")
        return render_template("hw2/measurement_step.html", calib=calib, image_url=image_url,
                                image_name=f.filename, result=None)

    image_name = request.form.get("image_name")
    distance_mm = float(request.form.get("distance_mm"))
    x1, y1 = float(request.form["x1"]), float(request.form["y1"])
    x2, y2 = float(request.form["x2"]), float(request.form["y2"])

    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]

    p1_u, p2_u = dimensions.undistort_points([(x1, y1), (x2, y2)], camera_matrix, dist_coeffs)
    real_mm, dx_mm, dy_mm = dimensions.pixel_length_to_real(p1_u, p2_u, distance_mm, camera_matrix)
    pixel_dist = float(np.hypot(x2 - x1, y2 - y1))

    result = {
        "pixel_dist": pixel_dist,
        "distance_mm": distance_mm,
        "real_mm": real_mm,
        "real_cm": real_mm / 10.0,
    }

    # the photo with the measured segment drawn on it, for the result/sample
    img = cv2.imread(os.path.join(UPLOAD_OBJ_DIR, image_name))
    if img is not None:
        s = max(1, int(round(max(img.shape[:2]) / 900)))
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.line(img, p1, p2, (0, 255, 0), 2 * s, cv2.LINE_AA)
        for p in (p1, p2):
            cv2.circle(img, p, 6 * s, (0, 0, 255), -1)
        mid = ((p1[0] + p2[0]) // 2 + 10 * s, (p1[1] + p2[1]) // 2 - 10 * s)
        label = f"{real_mm / 10:.2f} cm"
        cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (0, 0, 0), 6 * s, cv2.LINE_AA)
        cv2.putText(img, label, mid, cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, (0, 255, 255), 2 * s, cv2.LINE_AA)
        fn = f"{int(time.time() * 1000)}_measurement.jpg"
        cv2.imwrite(os.path.join(RESULTS_DIR, fn), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        result["annotated_url"] = url_for("hw2.static", filename=f"results/{fn}")

    image_url = url_for("hw2.static", filename=f"object_uploads/{image_name}")
    offer = samples.offer("hw2", "measurement", {"result": result})
    return render_template("hw2/measurement_step.html", calib=calib, image_url=image_url,
                            image_name=image_name, result=result, sample_offer=offer)


@hw2.route("/validate", methods=["GET", "POST"])
def validate_view():
    if request.method == "GET":
        sample = samples.load("hw2", "validation")
        if sample:
            return render_template("hw2/validation_step.html", stats=sample["stats"], table=sample["table"],
                                   plot1=sample["plot1"], plot2=sample["plot2"], sample_info=sample["_sample"])
        return render_template("hw2/validation_step.html", stats=None, table=None, plot1=None, plot2=None)

    f = request.files.get("csv_file")
    if not f or not f.filename:
        flash("Choose a CSV file first.")
        return redirect(url_for("hw2.validate_view"))

    import pandas as pd
    df = pd.read_csv(f)
    required_cols = {"object_id", "actual_mm", "measured_mm"}
    if not required_cols.issubset(df.columns):
        flash(f"CSV needs at least these columns: {sorted(required_cols)}")
        return redirect(url_for("hw2.validate_view"))

    df_with_error, stats = vm.compute_stats(df)
    vm.make_plots(df_with_error, RESULTS_DIR)

    table_html = df_with_error.round(2).to_html(index=False, classes="result-table")
    # per-run copies, so saving this run as a sample later can't pick up a newer run's plots
    stamp = int(time.time() * 1000)
    for name in ("error_by_object", "percent_error"):
        shutil.copy(os.path.join(RESULTS_DIR, f"{name}.png"), os.path.join(RESULTS_DIR, f"{stamp}_{name}.png"))
    plot1 = url_for("hw2.static", filename=f"results/{stamp}_error_by_object.png")
    plot2 = url_for("hw2.static", filename=f"results/{stamp}_percent_error.png")
    df_with_error.round(3).to_csv(os.path.join(RESULTS_DIR, f"{stamp}_validation.csv"), index=False)

    stats = {k: (v.item() if hasattr(v, "item") else v) for k, v in stats.items()}
    offer = samples.offer("hw2", "validation", {
        "stats": stats, "table": table_html, "plot1": plot1, "plot2": plot2,
        "csv": url_for("hw2.static", filename=f"results/{stamp}_validation.csv")})
    return render_template("hw2/validation_step.html", stats=stats, table=table_html, plot1=plot1, plot2=plot2,
                           sample_offer=offer)
