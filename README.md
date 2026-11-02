# CSc 8830 Computer Vision — Assignments

Hosted on: https://computervision-homeworks.onrender.com/

## Challenge 1 — Context-Aware Hierarchical Robot Control

Notebook: [`Challenge1/challenge1_prototype.ipynb`](Challenge1/challenge1_prototype.ipynb) (run in Colab on a GPU).
A quick feasibility test showing that a frozen Video-LLM (Qwen2.5-VL) can understand an unseen home from a short video and pass the robot structured context: Perception ➔ Reasoning ➔ Environment Grounding ➔ Robot Action. It compares the robot with and without that context across different homes and lighting.

## Run

```bash
python -m venv venv
source venv/bin/activate        # venv\Scripts\activate on Windows
pip install -r requirements.txt
python main.py
```

Open `http://127.0.0.1:8000`.

## Offline samples (view results without uploading)

Every homework has a **Sample Output** entry in its nav menu, and every module page has a **Run / Sample Output** tab bar at the top, that shows a finished run with nothing to upload, while the
homework's own pages stay as clean forms for running new experiments.

1. Run the app locally (`python main.py`) and run a module with your own photos/videos.
2. Under the result, click **Save this run as the offline sample**. It copies the run's output files and
   numbers into `Homework/<hw>/static/samples/<name>/` (this replaces any earlier sample of that module).
3. Commit the `samples/` folders and deploy. The Sample Output tab now shows that run, with a
   **Download sample (ZIP)** link. Until a run is saved, it shows the repo's example output where there is one.

| Run page | Sample(s), shown at `/<hw>/sample-output` |
|---|---|
| `/hw2/calibrate`, `/hw2/measure`, `/hw2/validate` | `calibration`, `measurement`, `validation` |
| `/hw3/` | `blur` (both kernels) |
| `/hw4/` | `rgb`, `thermal` |
| `/hw5/flow` | `flow_video_1`, `flow_video_2` |
| `/hw5/tracking` | `track_video_1`, `track_video_2` (one per video's frame pair) |
| `/hw5/sfm` | `sfm` |
| `/hw7/stereo` | `stereo` |
| `/hw7/room` | `room` |

Saving and deleting only work when running with `python main.py` (or with `ALLOW_SAMPLE_SAVE=1`), so visitors
to the hosted site can't overwrite your samples; there, samples are view/download only. The hosted server's disk
is reset on every deploy, so samples must be committed to show up there. The HW5 flow samples include the
processed video (~20 MB for 30 s), so all HW5 samples together add roughly 60-80 MB to the repo.

## Challenge 1

`/challenge1/` shows the saved results of `Challenge1/challenge1_prototype.ipynb`: the summary, per-run table, figure,
and each home's scene context, plan and actions. The page reads them from the notebook's outputs. The notebook itself
ran on a Colab GPU, so nothing runs on the site, and it needs to be deployed alongside the app.

## HW2

Camera calibration, then turning a pixel measurement into a real-world length, then checking how accurate that is over a batch of objects.

**Working demo:** `/hw2/calibrate` (upload chessboard photos), `/hw2/measure` (upload an object photo, click two
points), `/hw2/validate` (upload a CSV of actual vs. measured lengths). Each of the three pages also ships an
**offline example output** below its form/upload button — corner detection on one of the repo's own chessboard
photos, a measured example on a real object photo, and the two error plots generated from the repo's own
`result.csv` — so the result is visible before you upload anything.

The latest calibration is saved in `static/calib/camera_calib.npz` and shown under **Saved camera intrinsics** on
`/hw2/calibrate` until the next run replaces it. HW7 reads the same file. Commit it after recalibrating so the hosted
site has the new values.

## HW3

Blurs an image in the spatial domain and through the FFT, then checks the two results actually match.

**Working demo:** `/hw3/` — upload an image button, runs both a Gaussian and a box kernel and shows the
side-by-side comparison panel. The page also shows an **offline example output**: the same panel pre-generated
from a sample photo in the repo, for both kernels.

## HW4

Pulls a human's outline out of an RGB photo and a thermal photo using classic OpenCV, and compares it against SAM2.

**Working demo:** `/hw4/` — separate upload-an-image buttons for the RGB and thermal pipelines, with an optional
SAM2 mask upload to score agreement against. The RGB section also shows an **offline example output**: a real
run of the pipeline (full mask + SAM2 IoU/Dice agreement), with the subject's face blurred before it was
committed. Thermal has no offline example yet — every thermal photo tried so far turned out to be a
regular photo, not real thermal-camera data, so upload your own to see that pipeline run.

## HW5

Optical flow + object tracking on two real videos, the Lucas-Kanade tracking equations derived from brightness
constancy and validated against independently measured pixel locations, a from-scratch bilinear interpolation,
and a 4-viewpoint structure-from-motion reconstruction of a planar object's boundary. `/hw5/` walks through the
whole workflow and maps every assignment requirement to where it is answered.

**Working demo:**
- `/hw5/flow`: two video slots. Each upload runs in the background (live preview while it processes) and
  produces a real-time 2x2 WebM video: moving objects boxed with IDs/speed/direction plus KLT feature trails, HSV
  flow, flow vectors, and camera-compensated motion. It also writes per-frame stats, an object-track table, plots
  and a "what can be inferred" summary backed by the numbers.
- `/hw5/tracking`: one click from each video's results hands over two consecutive full-resolution frames. Our
  pyramidal LK is compared with template-matching (NCC) locations and OpenCV LK, with a worked numeric example of
  the LK iterations and of bilinear interpolation.
- `/hw5/sfm`: upload 4 photos of a flat object and click its corners + extra points. You get the camera
  intrinsics (HW2 calibration / EXIF focal length / 60 deg FOV fallback), each camera's position, the full
  homography -> pose -> triangulation workout, and the estimated boundary.

Every module has a **Download all outputs (ZIP)** button (`/hw5/download/<run>.zip`) to keep its sample output.
Once you save offline samples (see above), each page shows them without any upload.

## HW7

Real-world measurement from stereo geometry and from video. Part 2 is a handwritten derivation, so it has no page.
`/hw7/` is the overview.

**Working demo:**
- `/hw7/stereo`: upload a left and right photo with the baseline and a calibration (HW2, EXIF or typed in). The app
  estimates the relative pose, rectifies, and shows the disparity and depth maps. Click object points on the rectified
  left image to measure a rectangle (width, height), circle (diameter) or polygon (every edge) from their 3D positions.
- `/hw7/room`: upload one room video. The app extracts key frames and runs incremental structure from motion.
  The room's corners and its floor and ceiling lines are found automatically, and the focal length is corrected from
  the room's right angles. Width is reported as a lower limit unless both ends of the front wall appear. Corners can
  also be marked by hand on two or more frames, and the scale can come from the camera's travel, the camera-to-wall distance or
  a known length. It reports length, width and height, with the unused references as independent checks. If the video
  shows no back wall, length is the camera-to-front-wall distance, which assumes the camera was at the back wall.

The form defaults are this homework's capture measurements: 297 mm baseline and 1188 mm object distance for stereo,
1260 mm left-right travel and 2673 mm camera-to-wall distance for video. Change them for other captures.

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
                         templates/hw4/, static/{rgb_uploads,thermal_uploads,sam2_uploads,outputs,demo}/
  hw5_motion_sfm/        routes.py, optical_flow.py, tracking.py, bilinear.py, sfm_planar.py, reports.py
                         templates/hw5/, static/{video_uploads,sfm_uploads,outputs/<run_id>/}
  hw_7/                  routes.py, stereo_measure.py, room_video.py, room_auto.py
                         templates/hw7/, static/{stereo_uploads,room_uploads,outputs/<run_id>/}, static/picker.js
```
