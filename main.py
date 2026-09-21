# entry point for the whole thing - just wires up the three homework
# pieces and serves an overview page linking to all of them

from flask import Flask, render_template

from Homework.hw2_calibration import hw2
from Homework.hw2_calibration.routes import get_current_calibration
from Homework.hw3_fourier_blur import hw3
from Homework.hw4_silhouette import hw4


def create_app():
    app = Flask(__name__)
    app.secret_key = "csc8830-cv-hub-dev-key"  # fine for a local class demo, not for production

    app.register_blueprint(hw2)
    app.register_blueprint(hw3)
    app.register_blueprint(hw4)

    @app.route("/")
    def overview():
        return render_template("overview.html", calib=get_current_calibration())

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8000)
