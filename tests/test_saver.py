import time
import numpy as np
from pathlib import Path
from src.saver import SaveDecider, save_frame, save_crop, save_label


def test_new_track_should_save():
    sd = SaveDecider(vehicle_cooldown=3.0, plate_cooldown=2.0, save_better_only=True)
    track = {
        "track_id": "veh_0", "object_type": "vehicle",
        "last_saved_ts": None, "save_count": 0,
        "best_quality_score": 0.0, "is_new": True,
    }
    assert sd.should_save(track, quality_score=0.5) is True


def test_cooldown_blocks_save():
    sd = SaveDecider(vehicle_cooldown=3.0, plate_cooldown=2.0, save_better_only=True)
    now = time.time()
    track = {
        "track_id": "veh_0", "object_type": "vehicle",
        "last_saved_ts": now, "save_count": 1,
        "best_quality_score": 0.5, "is_new": False,
    }
    assert sd.should_save(track, quality_score=0.5) is False


def test_better_quality_overrides_cooldown():
    sd = SaveDecider(vehicle_cooldown=3.0, plate_cooldown=2.0, save_better_only=True)
    now = time.time()
    track = {
        "track_id": "veh_0", "object_type": "vehicle",
        "last_saved_ts": now, "save_count": 1,
        "best_quality_score": 0.3, "is_new": False,
    }
    assert sd.should_save(track, quality_score=0.8) is True


def test_save_frame_creates_file(tmp_path):
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    path = save_frame(frame, tmp_path, "sess_test", 1000, 5)
    assert path.exists()
    assert path.suffix == ".jpg"


def test_save_crop_creates_png(tmp_path):
    crop = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    path = save_crop(crop, tmp_path, "sess_test", "veh_0", 1000, "vehicle")
    assert path.exists()
    assert path.suffix == ".png"


def test_save_label_creates_txt(tmp_path):
    detections = [
        {"object_type": "vehicle", "bbox_xyxy": [100, 100, 400, 300]},
        {"object_type": "plate", "bbox_xyxy": [200, 250, 350, 290]},
    ]
    path = save_label(detections, 1920, 1080, tmp_path, "sess_test", 1000, 5)
    assert path.exists()
    lines = path.read_text().strip().split("\n")
    assert len(lines) == 2
    parts = lines[0].split()
    assert parts[0] == "0"  # vehicle class id
    assert len(parts) == 5  # class x_center y_center w h
