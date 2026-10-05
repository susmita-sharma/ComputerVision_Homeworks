# entry point for the whole thing - just wires up the three homework
# pieces and serves an overview page linking to all of them

import os

from flask import Flask, render_template

from Homework.hw2_calibration import hw2
from Homework.hw2_calibration.routes import get_current_calibration
from Homework.hw3_fourier_blur import hw3
from Homework.hw4_silhouette import hw4
from Homework.hw5_motion_sfm import hw5
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
    app.register_blueprint(samples_bp)

    @app.route("/")
    def overview():
        return render_template("overview.html", calib=get_current_calibration())

    return app


app = create_app()

if __name__ == "__main__":
    app.config["ALLOW_SAMPLE_SAVE"] = os.environ.get("ALLOW_SAMPLE_SAVE", "1") == "1"
    app.run(debug=True, host="0.0.0.0", port=8000)
