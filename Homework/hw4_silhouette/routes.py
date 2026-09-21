# web glue for HW4 - two forms (RGB, thermal), each running its own
# segmentation pipeline and optionally scoring against an uploaded SAM2 mask

import os
import time

import cv2
from flask import Blueprint, render_template, request, redirect, url_for, flash

from Homework import STATIC_ROOT
from . import rgb_boundary, thermal_boundary, mask_metrics

hw4 = Blueprint("hw4", __name__, url_prefix="/hw4")

RGB_UPLOAD_DIR = os.path.join(STATIC_ROOT, "hw4", "rgb_uploads")
THERMAL_UPLOAD_DIR = os.path.join(STATIC_ROOT, "hw4", "thermal_uploads")
SAM2_UPLOAD_DIR = os.path.join(STATIC_ROOT, "hw4", "sam2_uploads")
OUTPUT_DIR = os.path.join(STATIC_ROOT, "hw4", "outputs")
for d in (RGB_UPLOAD_DIR, THERMAL_UPLOAD_DIR, SAM2_UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)

MAX_DIM = 800


def _load_resized(path):
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None
    h, w = image.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))
    return image


def _save_png(array_uint8, tag, stamp):
    filename = f"{stamp}_{tag}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), array_uint8)
    return url_for("static", filename=f"hw4/outputs/{filename}")


def _run_pipeline(image, seg, section, stamp, sam2_file):
    urls = {
        "original_url": _save_png(image, f"{section}_original", stamp),
        "gray_url": _save_png(seg["gray"], f"{section}_gray", stamp),
        "otsu_url": _save_png(seg["otsu_mask"], f"{section}_otsu", stamp),
        "cleaned_url": _save_png(seg["cleaned_mask"], f"{section}_cleaned", stamp),
        "boundary_mask_url": _save_png(seg["boundary_mask"], f"{section}_boundary_mask", stamp),
        "overlay_url": _save_png(seg["overlay"], f"{section}_overlay", stamp),
    }

    comparison = None
    if sam2_file and sam2_file.filename:
        sam2_path = os.path.join(SAM2_UPLOAD_DIR, f"{stamp}_{sam2_file.filename}")
        sam2_file.save(sam2_path)
        sam2_mask = mask_metrics.load_binary_mask(sam2_path, seg["boundary_mask"].shape)
        metrics = mask_metrics.compare_masks(seg["boundary_mask"], sam2_mask)
        overlay_diff = mask_metrics.agreement_overlay(seg["boundary_mask"], sam2_mask)
        comparison = {
            "metrics": metrics,
            "sam2_mask_url": _save_png(sam2_mask, f"{section}_sam2_mask", stamp),
            "agreement_url": _save_png(overlay_diff, f"{section}_agreement", stamp),
        }

    return urls, comparison


@hw4.route("/", methods=["GET"])
def home():
    return render_template("hw4/boundary_extraction.html", rgb_result=None, thermal_result=None)


@hw4.route("/rgb", methods=["POST"])
def process_rgb():
    f = request.files.get("image")
    if not f or not f.filename:
        flash("Choose an RGB image first.")
        return redirect(url_for("hw4.home"))

    stamp = str(int(time.time() * 1000))
    in_path = os.path.join(RGB_UPLOAD_DIR, f"{stamp}_{f.filename}")
    f.save(in_path)

    image = _load_resized(in_path)
    if image is None:
        flash("Could not read that file as an image.")
        return redirect(url_for("hw4.home"))

    seg = rgb_boundary.segment_human_rgb(image)
    urls, comparison = _run_pipeline(image, seg, "rgb", stamp, request.files.get("sam2_mask"))

    rgb_result = {**urls, "comparison": comparison}
    return render_template("hw4/boundary_extraction.html", rgb_result=rgb_result, thermal_result=None)


@hw4.route("/thermal", methods=["POST"])
def process_thermal():
    f = request.files.get("image")
    if not f or not f.filename:
        flash("Choose a thermal image first.")
        return redirect(url_for("hw4.home"))

    hot_is_bright = request.form.get("hot_is_bright", "yes") == "yes"

    stamp = str(int(time.time() * 1000))
    in_path = os.path.join(THERMAL_UPLOAD_DIR, f"{stamp}_{f.filename}")
    f.save(in_path)

    image = _load_resized(in_path)
    if image is None:
        flash("Could not read that file as an image.")
        return redirect(url_for("hw4.home"))

    seg = thermal_boundary.segment_human_thermal(image, hot_is_bright=hot_is_bright)
    urls, comparison = _run_pipeline(image, seg, "thermal", stamp, request.files.get("sam2_mask"))

    thermal_result = {**urls, "comparison": comparison}
    return render_template("hw4/boundary_extraction.html", rgb_result=None, thermal_result=thermal_result)
