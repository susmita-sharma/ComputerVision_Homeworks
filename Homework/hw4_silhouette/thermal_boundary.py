# same idea as rgb_boundary.py, but for thermal photos. the pipeline itself
# (Otsu -> cleanup -> watershed -> contour) doesn't change, only how we decide
# which side of the threshold is the person: in most thermal palettes a warm
# body shows up brighter than the background, so I use mean intensity instead
# of the border trick from the RGB version.

import cv2

from . import segmentation_core as core


def segment_human_thermal(image_bgr, hot_is_bright=True):
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr.copy()
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    inverted = cv2.bitwise_not(otsu)

    mean_otsu = cv2.mean(gray, mask=otsu)[0]
    mean_inverted = cv2.mean(gray, mask=inverted)[0]

    # some palettes map hot to dark instead of bright, hence the toggle
    if hot_is_bright:
        raw_mask = otsu if mean_otsu >= mean_inverted else inverted
    else:
        raw_mask = otsu if mean_otsu <= mean_inverted else inverted

    color_for_overlay = image_bgr if image_bgr.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    result = core.run_classical_pipeline(gray, color_for_overlay, raw_mask)
    result["gray"] = gray
    return result
