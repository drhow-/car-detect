import cv2
import numpy as np


def _blur_score(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _truncation_fraction(bbox_xyxy, image_shape):
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    cx1 = max(0, x1)
    cy1 = max(0, y1)
    cx2 = min(w, x2)
    cy2 = min(h, y2)
    full_area = (x2 - x1) * (y2 - y1)
    if full_area <= 0:
        return 1.0
    clipped_area = max(0, cx2 - cx1) * max(0, cy2 - cy1)
    return 1.0 - (clipped_area / full_area)


def _exposure_ok(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    return 20 < mean_val < 240


def check_quality(crop, object_type, bbox_xyxy, image_shape,
                  min_plate_w=40, min_plate_h=14,
                  min_vehicle_area=12000, min_blur_score=50.0):
    h, w = crop.shape[:2]
    blur = _blur_score(crop)
    trunc_frac = _truncation_fraction(bbox_xyxy, image_shape)
    truncated = trunc_frac > 0.05
    exposure = _exposure_ok(crop)

    if object_type == "plate":
        if w < min_plate_w or h < min_plate_h:
            return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                    "reject_reason": "size_too_small", "truncated": truncated}
    else:
        if w * h < min_vehicle_area:
            return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                    "reject_reason": "size_too_small", "truncated": truncated}

    if trunc_frac > 0.30:
        return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                "reject_reason": "truncated", "truncated": True}

    if blur < min_blur_score:
        return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                "reject_reason": "too_blurry", "truncated": truncated}

    if not exposure:
        return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                "reject_reason": "bad_exposure", "truncated": truncated}

    blur_norm = min(blur / 500.0, 1.0)
    size_norm = min((w * h) / 100000.0, 1.0) if object_type == "vehicle" else min(w / 200.0, 1.0)
    trunc_penalty = 1.0 - trunc_frac
    quality_score = 0.4 * blur_norm + 0.35 * size_norm + 0.25 * trunc_penalty

    return {"passes": True, "quality_score": quality_score, "blur_score": blur,
            "reject_reason": None, "truncated": truncated}
