# Challenge 1 - shows the saved results of the Colab notebook. Nothing runs here:
# it needs a GPU and the Qwen2.5-VL and OWLv2 models.

import io

from flask import Blueprint, render_template, send_file

from . import notebook

challenge1 = Blueprint("challenge1", __name__, url_prefix="/challenge1", template_folder="templates")


@challenge1.route("/")
def home():
    results, figure = notebook.load()
    return render_template("challenge1/samples.html", results=results, has_figure=figure is not None)


@challenge1.route("/figure.png")
def figure():
    return send_file(io.BytesIO(notebook.load()[1]), mimetype="image/png")


@challenge1.route("/challenge1_prototype.ipynb")
def download():
    return send_file(notebook.NOTEBOOK, as_attachment=True)
