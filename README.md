# CSc 8830 Computer Vision — Assignments
## Hosted on: https://computervision-homeworks.onrender.com/

## Run

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:8000`.

## HW2

Camera calibration, then turning a pixel measurement into a real-world length, then checking how accurate that is over a batch of objects.

## HW3

Blurs an image in the spatial domain and through the FFT, then checks the two results actually match.

## HW4

Pulls a human's outline out of an RGB photo and a thermal photo using classic OpenCV, and compares it against SAM2.

## Layout

```
main.py
Homework/   hw2_calibration/, hw3_fourier_blur/, hw4_silhouette/
templates/  shell.html, overview.html, hw2/, hw3/, hw4/
static/     calib*/, object_uploads/, results/, hw3/, hw4/
```
