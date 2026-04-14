import json
from pathlib import Path
from src.metadata import MetadataLogger


def test_log_save_event(tmp_path):
    logger = MetadataLogger(tmp_path)
    logger.log_save_event({
        "event_id": "evt_000001",
        "session_id": "sess_test",
        "timestamp": "2026-04-14T14:30:22.481Z",
        "camera_id": "cam0",
        "frame_path": "raw/frames/test.jpg",
        "label_path": "raw/labels/test.txt",
        "image_width": 1920,
        "image_height": 1080,
        "detections": [],
    })
    log_path = tmp_path / "raw" / "metadata" / "save_events.jsonl"
    assert log_path.exists()
    line = json.loads(log_path.read_text().strip())
    assert line["event_id"] == "evt_000001"


def test_log_track_lifecycle(tmp_path):
    logger = MetadataLogger(tmp_path)
    logger.log_track({
        "track_id": "veh_0",
        "object_type": "vehicle",
        "first_seen_ts": 1000.0,
        "last_seen_ts": 1005.0,
        "save_count": 2,
    })
    log_path = tmp_path / "raw" / "metadata" / "tracks.jsonl"
    assert log_path.exists()


def test_log_session(tmp_path):
    logger = MetadataLogger(tmp_path)
    logger.log_session({
        "session_id": "sess_test",
        "start_time": "2026-04-14T14:00:00Z",
        "end_time": "2026-04-14T14:30:00Z",
        "total_frames": 5000,
        "total_saved": 42,
    })
    log_path = tmp_path / "raw" / "metadata" / "sessions.jsonl"
    assert log_path.exists()


def test_event_counter_increments(tmp_path):
    logger = MetadataLogger(tmp_path)
    eid1 = logger.next_event_id()
    eid2 = logger.next_event_id()
    assert eid1 == "evt_000001"
    assert eid2 == "evt_000002"


def test_classes_txt_created(tmp_path):
    logger = MetadataLogger(tmp_path)
    logger.ensure_classes_txt()
    classes_path = tmp_path / "raw" / "classes.txt"
    assert classes_path.exists()
    assert classes_path.read_text() == "vehicle\nplate\n"
