import time
from src.tracker import Tracker


def _det(x1, y1, x2, y2, obj_type="vehicle", conf=0.8):
    return {
        "bbox_xyxy": [x1, y1, x2, y2],
        "conf": conf,
        "class_name": "car" if obj_type == "vehicle" else "license_plate",
        "object_type": obj_type,
    }


def test_new_detection_creates_track():
    t = Tracker(track_timeout=4.0)
    tracks = t.update([_det(100, 100, 300, 300)])
    assert len(tracks) == 1
    assert tracks[0]["track_id"].startswith("veh_")


def test_plate_track_prefix():
    t = Tracker(track_timeout=4.0)
    tracks = t.update([_det(100, 100, 200, 130, "plate")])
    assert tracks[0]["track_id"].startswith("plt_")


def test_same_object_reuses_track():
    t = Tracker(track_timeout=4.0)
    t.update([_det(100, 100, 300, 300)])
    tracks = t.update([_det(105, 105, 305, 305)])  # slight shift
    assert len(tracks) == 1


def test_stale_track_removed():
    t = Tracker(track_timeout=0.01)  # very short timeout for test
    t.update([_det(100, 100, 300, 300)])
    time.sleep(0.02)
    tracks = t.update([])  # no detections
    assert len(tracks) == 0
    assert len(t.active_tracks) == 0


def test_two_separate_objects_two_tracks():
    t = Tracker(track_timeout=4.0)
    tracks = t.update([
        _det(100, 100, 200, 200),
        _det(500, 500, 700, 700),
    ])
    assert len(tracks) == 2
    ids = {tr["track_id"] for tr in tracks}
    assert len(ids) == 2
