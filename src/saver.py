import time
from datetime import datetime
from pathlib import Path
import cv2
import numpy as np

_CLASS_MAP = {"vehicle": 0, "plate": 1}


class SaveDecider:
    BETTER_THRESHOLD = 0.2

    def __init__(self, vehicle_cooldown=3.0, plate_cooldown=2.0, save_better_only=True):
        self.cooldowns = {"vehicle": vehicle_cooldown, "plate": plate_cooldown}
        self.save_better_only = save_better_only

    def should_save(self, track, quality_score):
        if track["is_new"] or track["save_count"] == 0:
            return True
        cooldown = self.cooldowns.get(track["object_type"], 3.0)
        now = time.time()
        elapsed = now - (track["last_saved_ts"] or 0)
        if self.save_better_only and quality_score > track["best_quality_score"] + self.BETTER_THRESHOLD:
            return True
        if elapsed >= cooldown:
            return True
        return False


def _date_subdir():
    return datetime.now().strftime("%Y-%m-%d")


def save_frame(frame, output_dir, session_id, timestamp_ms, frame_idx):
    subdir = Path(output_dir) / "raw" / "frames" / _date_subdir()
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"frm_{session_id}_{timestamp_ms}_{frame_idx:06d}.jpg"
    path = subdir / fname
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def save_crop(crop, output_dir, session_id, track_id, timestamp_ms, object_type):
    kind = "vehicles" if object_type == "vehicle" else "plates"
    prefix = "veh" if object_type == "vehicle" else "plt"
    subdir = Path(output_dir) / "raw" / "crops" / kind / _date_subdir()
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{session_id}_{track_id}_{timestamp_ms}.png"
    path = subdir / fname
    cv2.imwrite(str(path), crop)
    return path


def save_label(detections, img_w, img_h, output_dir, session_id, timestamp_ms, frame_idx):
    subdir = Path(output_dir) / "raw" / "labels" / _date_subdir()
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"frm_{session_id}_{timestamp_ms}_{frame_idx:06d}.txt"
    path = subdir / fname
    lines = []
    for det in detections:
        cls_id = _CLASS_MAP.get(det["object_type"], 0)
        x1, y1, x2, y2 = det["bbox_xyxy"]
        x_center = ((x1 + x2) / 2) / img_w
        y_center = ((y1 + y2) / 2) / img_h
        w = (x2 - x1) / img_w
        h = (y2 - y1) / img_h
        lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    path.write_text("\n".join(lines) + "\n")
    return path


def extract_crop(frame, bbox_xyxy, padding_frac=0.1):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox_xyxy]
    bw, bh = x2 - x1, y2 - y1
    pad_x = int(bw * padding_frac)
    pad_y = int(bh * padding_frac)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(w, x2 + pad_x)
    cy2 = min(h, y2 + pad_y)
    return frame[cy1:cy2, cx1:cx2].copy()
