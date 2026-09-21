import os
import time

import cv2
import numpy as np
from flask import Blueprint, render_template, request, redirect, url_for, flash

from Homework import STATIC_ROOT
from . import spectral_blur as sb

hw3 = Blueprint("hw3", __name__, url_prefix="/hw3")

UPLOAD_DIR = os.path.join(STATIC_ROOT, "hw3", "uploads")
OUTPUT_DIR = os.path.join(STATIC_ROOT, "hw3", "outputs")
for d in (UPLOAD_DIR, OUTPUT_DIR):
    os.makedirs(d, exist_ok=True)


def _save_png(array_uint8, name, stamp):
    filename = f"{stamp}_{name}.png"
    cv2.imwrite(os.path.join(OUTPUT_DIR, filename), array_uint8)
    return url_for("static", filename=f"hw3/outputs/{filename}")


@hw3.route("/", methods=["GET", "POST"])
def blur_view():
    if request.method == "GET":
        return render_template("hw3/frequency_blur.html", result=None)

    f = request.files.get("image")
    if not f or not f.filename:
        flash("Choose an image first.")
        return redirect(url_for("hw3.blur_view"))

    kernel_type = request.form.get("kernel_type", "gaussian")
    ksize = int(request.form.get("ksize", 15))
    sigma = float(request.form.get("sigma", 0) or 0)

    in_path = os.path.join(UPLOAD_DIR, f.filename)
    f.save(in_path)

    image = cv2.imread(in_path, cv2.IMREAD_COLOR)
    if image is None:
        flash("Could not read that file as an image.")
        return redirect(url_for("hw3.blur_view"))

    # don't let someone upload a giant phone photo and grind the FFT to a halt
    max_dim = 700
    h, w = image.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        image = cv2.resize(image, (int(w * scale), int(h * scale)))

    blurred = sb.run_both_domains(image, kernel_type, ksize, sigma if sigma > 0 else None)

    diff_gray = cv2.cvtColor(
        cv2.absdiff(blurred["spatial_result"], blurred["freq_result"]), cv2.COLOR_BGR2GRAY
    ) if image.ndim == 3 else cv2.absdiff(blurred["spatial_result"], blurred["freq_result"])
    # the raw diff is basically all black (that's the whole point), so bump
    # the contrast way up just so there's something visible to look at
    diff_boosted = np.clip(diff_gray.astype(np.float64) * 40.0, 0, 255).astype(np.uint8)

    comparison_panel = sb.build_comparison_panel([
        ("original", image),
        ("spatial blur", blurred["spatial_result"]),
        ("frequency blur", blurred["freq_result"]),
        ("difference x40", diff_boosted),
    ])

    image_spectrum_img = sb.log_magnitude_spectrum(blurred["image_spectrum"])
    kernel_spectrum_img = sb.log_magnitude_spectrum(blurred["kernel_spectrum"])
    spectrum_panel = sb.build_comparison_panel([
        ("image spectrum", image_spectrum_img),
        ("kernel spectrum", kernel_spectrum_img),
    ])

    stamp = str(int(time.time() * 1000))
    comparison_url = _save_png(comparison_panel, "comparison", stamp)
    spectrum_url = _save_png(spectrum_panel, "spectrum", stamp)

    result = {
        "comparison_url": comparison_url,
        "spectrum_url": spectrum_url,
        "metrics": blurred["metrics"],
        "kernel_type": kernel_type,
        "ksize": blurred["metrics"]["kernel_shape"][0],
        "sigma": sigma if sigma > 0 else round(blurred["metrics"]["kernel_shape"][0] / 6.0, 3),
    }
    return render_template("hw3/frequency_blur.html", result=result)
