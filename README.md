# CSc 8830 Computer Vision — Assignments
<<<<<<< HEAD
## Hosted on: https://computervision-homeworks.onrender.com/
=======
>>>>>>> 18cb6ad (hw4)

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

`/hw2/calibrate` (upload chessboard photos), `/hw2/measure` (upload an object photo, click two
points), `/hw2/validate` (upload a CSV of actual vs. measured lengths). Each of the three pages also ships an
**offline example output** below its form/upload button — corner detection on one of the repo's own chessboard
photos, a measured example on a real object photo, and the two error plots generated from the repo's own

## HW3

Blurs an image in the spatial domain and through the FFT, then checks the two results actually match.

`/hw3/` — upload an image button, runs both a Gaussian and a box kernel and shows the
side-by-side comparison panel. 

## HW4

Pulls a human's outline out of an RGB photo and a thermal photo using classic OpenCV, and compares it against SAM2.

`/hw4/` — separate upload-an-image buttons for the RGB and thermal pipelines, with an optional
SAM2 mask upload to score agreement against. No RGB/thermal/SAM2 sample photos ship in the repo yet, so there's
no offline example output here (yet) — upload your own photo to see it run.

## HW5

Optical flow on a real video, the Lucas-Kanade tracking equations derived from brightness constancy and
validated against OpenCV on a real frame pair, a from-scratch bilinear interpolation, and a 4-viewpoint
structure-from-motion reconstruction of a planar object's boundary.

`/hw5/flow` (upload a video), `/hw5/tracking` (upload/hand off a frame pair, click points to
track), `/hw5/sfm` (upload 4 photos of a flat object, click corners + boundary points). No sample video/photos
ship in the repo yet, so there's no offline example output here (yet) — upload your own to see it run.

## Layout

Each homework is under `Homework/hwN_*/` — its backend code, its own `templates/` and `static/`
(uploads + generated outputs + any committed demo assets), and for HW2 the raw calibration/measurement photos
it was built from. Only `main.py`, the shared `templates/` (site shell + overview page) and project-level files
(requirements, Dockerfile, tests) sit outside.

```
main.py
templates/              shell.html, overview.html (shared shell + landing page only)
tests/                  test_hw3_kernel_comparison.py, test_hw5_*.py
Homework/
  hw2_calibration/      routes.py, calibration.py, dimensions.py, validate.py
                         templates/hw2/, static/{calib,calib_uploads,object_uploads,results,demo}/
                         data/{calibration_images,object_images,result.csv}   (real inputs used above)
  hw3_fourier_blur/      routes.py, spectral_blur.py
                         templates/hw3/, static/{uploads,outputs,demo}/
  hw4_silhouette/        routes.py, segmentation_core.py, rgb_boundary.py, thermal_boundary.py, mask_metrics.py
                         templates/hw4/, static/{rgb_uploads,thermal_uploads,sam2_uploads,outputs}/
  hw5_motion_sfm/        routes.py, optical_flow.py, tracking.py, bilinear.py, sfm_planar.py
                         templates/hw5/, static/{video_uploads,sfm_uploads,outputs}/
```
