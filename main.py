# entry point for the whole thing - just wires up the three homework
# pieces and serves an overview page linking to all of them

import os

from flask import Flask, render_template, url_for

from Homework.challenge1 import challenge1
from Homework.hw2_calibration import hw2
from Homework.hw3_fourier_blur import hw3
from Homework.hw4_silhouette import hw4
from Homework.hw5_motion_sfm import hw5
from Homework.hw_7 import hw7
from Homework.samples import samples_bp


def create_app():
    # no root-level static folder - each hwN_* package under Homework/ owns
    # its own templates/ and static/ (uploads + outputs), registered as its
    # blueprint's static_folder, so /hw2/static/..., /hw3/static/..., etc.
    app = Flask(__name__, static_folder=None)
    app.secret_key = "csc8830-cv-hub-dev-key"  # fine for a local class demo, not for production
    app.config["MAX_CONTENT_LENGTH"] = 300 * 1024 * 1024  # HW5 video uploads run bigger than the photo-only homeworks
    # "Save as offline sample" buttons: on for local `python main.py` runs (set
    # below), off on the hosted gunicorn deploy unless ALLOW_SAMPLE_SAVE=1
    app.config["ALLOW_SAMPLE_SAVE"] = os.environ.get("ALLOW_SAMPLE_SAVE") == "1"

    app.register_blueprint(hw2)
    app.register_blueprint(hw3)
    app.register_blueprint(hw4)
    app.register_blueprint(hw5)
    app.register_blueprint(hw7)
    app.register_blueprint(challenge1)
    app.register_blueprint(samples_bp)

    @app.route("/")
    def overview():
        tiles = [
            {"tag": "HW2", "title": "Camera Calibration & Measurement", "url": url_for("hw2.calibrate_view"),
             "text": "Chessboard calibration, real-world lengths from pixels, and an accuracy check."},
            {"tag": "HW3", "title": "Blur: Spatial vs. Frequency", "url": url_for("hw3.blur_view"),
             "text": "The same blur by convolution and by FFT, testing the convolution theorem."},
            {"tag": "HW4", "title": "Human Boundary Extraction", "url": url_for("hw4.home"),
             "text": "A person's outline from RGB and thermal photos with classical OpenCV, compared with SAM2."},
            {"tag": "HW5", "title": "Optical Flow, Tracking & Structure from Motion", "url": url_for("hw5.home"),
             "text": "Motion in real video, Lucas-Kanade tracking, and a 4-view reconstruction of a flat object."},
            {"tag": "HW7", "title": "Stereo Measurement & Room Dimensions", "url": url_for("hw7.home"),
             "text": "Object sizes from a calibrated stereo pair, and room dimensions from one video."},
            {"tag": "Challenge 1", "title": "Context-Aware Robot Control", "url": url_for("challenge1.home"),
             "text": "Saved results of a Video-LLM guiding a robot in unseen homes, without retraining."},
        ]
        return render_template("overview.html", tiles=tiles)

    return app


app = create_app()

if __name__ == "__main__":
    app.config["ALLOW_SAMPLE_SAVE"] = os.environ.get("ALLOW_SAMPLE_SAVE", "1") == "1"
    app.run(debug=True, host="0.0.0.0", port=8000)
