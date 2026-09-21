# pulls a human silhouette out of a normal RGB photo, no ML involved.
# grayscale -> blur -> Otsu threshold -> pick the right side -> hand off to
# the shared cleanup/watershed/contour pipeline in segmentation_core.py

import cv2

from . import segmentation_core as core


def segment_human_rgb(image_bgr):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(otsu)

    # Otsu just splits the image in two, it has no idea which half is the
    # person. Went with the border heuristic - the background usually eats
    # up more of the frame edge than the subject does.
    raw_mask = core.choose_foreground_by_border(otsu, inverted)

    result = core.run_classical_pipeline(gray, image_bgr, raw_mask)
    result["gray"] = gray
    return result
