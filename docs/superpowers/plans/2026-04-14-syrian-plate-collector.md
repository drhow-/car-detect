# Syrian License Plate Data Collector — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a real-time data collection pipeline that detects vehicles and license plates from a USB webcam, saves high-quality crops/labels/metadata for training a unified YOLO26 detector, vehicle brand classifier, and Arabic+Latin OCR model.

**Architecture:** Two YOLO detectors (YOLO26x for vehicles, YOLO11 for plates) run in parallel on every Nth frame. A tracking layer associates detections across frames, a quality gate filters junk, and a save engine writes crops (PNG), frames (JPEG 95%), pseudo-labels (YOLO format), and metadata (JSONL). Phase 1.5 adds AI review via Claude/Gemini vision APIs with grid batching. Phase 1.5b adds a human review CLI.

**Tech Stack:** Python 3.11+, ultralytics (YOLO26x + YOLO11), OpenCV, NumPy, anthropic/google-generativeai (Phase 1.5), Pillow

**Spec:** `docs/superpowers/specs/2026-04-14-syrian-plate-collector-design.md`

---

## File Structure

```text
src/
├── config.py              # CLI argument parsing + defaults (argparse)
├── capture.py             # Camera capture with frame skipping
├── detector.py            # Vehicle + plate detection (two YOLO models)
├── tracker.py             # Multi-object tracker (centroid+IoU+size)
├── association.py         # Vehicle-plate scored matching
├── quality.py             # Quality gate (blur, size, truncation, exposure)
├── saver.py               # Save decision engine + file I/O (crops, frames, labels)
├── metadata.py            # JSONL metadata logging (events, tracks, sessions)
├── display.py             # OpenCV live display with overlays
├── collector.py           # Main pipeline orchestrator
preflight.py               # Pre-flight plate model validation
review.py                  # AI review CLI (bbox, brand, plate modules)
human_review.py            # Human review CLI
requirements.txt           # Phase 1 + 1.5 dependencies
tests/
├── conftest.py            # Shared fixtures (fake frames, fake detections)
├── test_config.py         # Config parsing tests
├── test_tracker.py        # Tracker unit tests
├── test_association.py    # Vehicle-plate association tests
├── test_quality.py        # Quality gate tests
├── test_saver.py          # Save decision logic tests
├── test_metadata.py       # Metadata writing tests
├── test_capture.py        # Frame skipping tests
├── test_review.py         # AI review grid/parsing tests
└── test_plate_format.py   # Plate OCR hallucination guardrail tests
```

---

## Phase 1 — Collection Pipeline

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `src/__init__.py`
- Create: `src/config.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `tests/test_config.py`

- [ ] **Step 1: Create requirements.txt**

```text
ultralytics>=8.3.0
opencv-python>=4.9.0
numpy>=1.26.0
huggingface_hub>=0.20.0
```

- [ ] **Step 2: Create src/__init__.py and tests/__init__.py**

Both are empty files.

- [ ] **Step 3: Write test for config parsing**

```python
# tests/test_config.py
import sys
from src.config import parse_args


def test_defaults():
    args = parse_args([])
    assert args.camera == 0
    assert args.vehicle_model == "yolo26x.pt"
    assert args.plate_model == "morsetechlab/yolov11-license-plate-detection"
    assert args.vehicle_conf == 0.30
    assert args.plate_conf == 0.30
    assert args.imgsz == 960
    assert args.frame_stride == 3
    assert args.adaptive_stride is True
    assert args.proximity_margin_px == 60
    assert args.output == "./output"
    assert args.vehicle_cooldown == 3.0
    assert args.plate_cooldown == 2.0
    assert args.track_timeout == 4.0
    assert args.min_plate_w == 40
    assert args.min_plate_h == 14
    assert args.min_vehicle_area == 12000
    assert args.min_blur_score == 50.0
    assert args.save_better_only is True
    assert args.save_hard_negatives is True
    assert args.no_display is False
    assert args.session_id is None


def test_custom_args():
    args = parse_args(["--camera", "2", "--frame-stride", "5", "--no-display"])
    assert args.camera == 2
    assert args.frame_stride == 5
    assert args.no_display is True
```

- [ ] **Step 4: Run test to verify it fails**

Run: `python -m pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.config'`

- [ ] **Step 5: Implement config.py**

```python
# src/config.py
import argparse


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Syrian License Plate Data Collector")

    # Camera
    p.add_argument("--camera", type=int, default=0, help="Camera device index")

    # Models
    p.add_argument("--vehicle-model", default="yolo26x.pt", help="Vehicle detector model")
    p.add_argument("--plate-model", default="morsetechlab/yolov11-license-plate-detection",
                    help="Plate detector (HF repo id or local .pt path)")
    p.add_argument("--vehicle-conf", type=float, default=0.30, help="Vehicle detection threshold")
    p.add_argument("--plate-conf", type=float, default=0.30, help="Plate detection threshold")
    p.add_argument("--imgsz", type=int, default=960, help="Inference image size")

    # Frame skipping
    p.add_argument("--frame-stride", type=int, default=3,
                    help="Run detection on every Nth captured frame")
    p.add_argument("--adaptive-stride", type=lambda x: x.lower() != "false",
                    default=True, help="Auto-raise stride if inference can't keep up")

    # Association
    p.add_argument("--proximity-margin-px", type=int, default=60,
                    help="Pixels below vehicle bbox for provisional plate match")

    # Output
    p.add_argument("--output", default="./output", help="Output directory")

    # Cooldowns
    p.add_argument("--vehicle-cooldown", type=float, default=3.0,
                    help="Seconds between vehicle saves")
    p.add_argument("--plate-cooldown", type=float, default=2.0,
                    help="Seconds between plate saves")
    p.add_argument("--track-timeout", type=float, default=4.0,
                    help="Track stale timeout seconds")

    # Quality thresholds
    p.add_argument("--min-plate-w", type=int, default=40, help="Minimum plate width px")
    p.add_argument("--min-plate-h", type=int, default=14, help="Minimum plate height px")
    p.add_argument("--min-vehicle-area", type=int, default=12000, help="Minimum vehicle area px")
    p.add_argument("--min-blur-score", type=float, default=50.0, help="Blur rejection threshold")

    # Save behavior
    p.add_argument("--save-better-only", type=lambda x: x.lower() != "false",
                    default=True, help="Save again only if new crop is better")
    p.add_argument("--save-hard-negatives", type=lambda x: x.lower() != "false",
                    default=True, help="Keep useful negative examples")

    # Display
    p.add_argument("--no-display", action="store_true", help="Run headless")

    # Session
    p.add_argument("--session-id", default=None, help="Session identifier (auto-generated if omitted)")

    return p.parse_args(argv)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `python -m pytest tests/test_config.py -v`
Expected: 2 passed

- [ ] **Step 7: Create shared test fixtures**

```python
# tests/conftest.py
import numpy as np
import pytest


@pytest.fixture
def fake_frame_720p():
    """A 1280x720 random frame for testing."""
    return np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def fake_frame_1080p():
    """A 1920x1080 random frame for testing."""
    return np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8)


@pytest.fixture
def sample_vehicle_detection():
    """A single vehicle detection dict."""
    return {
        "bbox_xyxy": [120, 80, 450, 320],
        "conf": 0.81,
        "class_name": "car",
        "object_type": "vehicle",
    }


@pytest.fixture
def sample_plate_detection():
    """A single plate detection dict."""
    return {
        "bbox_xyxy": [200, 280, 340, 310],
        "conf": 0.77,
        "class_name": "license_plate",
        "object_type": "plate",
    }


@pytest.fixture
def output_dir(tmp_path):
    """A temporary output directory."""
    return tmp_path / "output"
```

- [ ] **Step 8: Commit**

```bash
git add requirements.txt src/__init__.py src/config.py tests/__init__.py tests/conftest.py tests/test_config.py
git commit -m "feat: project setup with config parsing and test fixtures"
```

---

### Task 2: Camera Capture with Frame Skipping

**Files:**
- Create: `src/capture.py`
- Create: `tests/test_capture.py`

- [ ] **Step 1: Write tests for frame skipping logic**

```python
# tests/test_capture.py
from src.capture import FrameGrabber


class FakeVideoCap:
    """Mock cv2.VideoCapture."""

    def __init__(self, frame_count=30):
        self._frame_count = frame_count
        self._idx = 0
        self._frame = __import__("numpy").zeros((720, 1280, 3), dtype=__import__("numpy").uint8)

    def isOpened(self):
        return self._idx < self._frame_count

    def read(self):
        if self._idx < self._frame_count:
            self._idx += 1
            return True, self._frame.copy()
        return False, None

    def get(self, prop_id):
        if prop_id == 5:  # CAP_PROP_FPS
            return 30.0
        if prop_id == 3:  # CAP_PROP_FRAME_WIDTH
            return 1280.0
        if prop_id == 4:  # CAP_PROP_FRAME_HEIGHT
            return 720.0
        return 0.0

    def release(self):
        pass


def test_frame_stride_skips_frames():
    cap = FakeVideoCap(frame_count=10)
    grabber = FrameGrabber(cap, frame_stride=3)
    results = []
    for _ in range(10):
        frame, should_process, idx = grabber.next()
        if frame is None:
            break
        results.append((should_process, idx))
    # Frames 0, 3, 6, 9 should be processed (every 3rd)
    process_indices = [idx for should, idx in results if should]
    assert process_indices == [0, 3, 6, 9]


def test_all_frames_returned_for_display():
    cap = FakeVideoCap(frame_count=6)
    grabber = FrameGrabber(cap, frame_stride=3)
    all_indices = []
    for _ in range(6):
        frame, _, idx = grabber.next()
        if frame is None:
            break
        all_indices.append(idx)
    assert all_indices == [0, 1, 2, 3, 4, 5]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_capture.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement capture.py**

```python
# src/capture.py
import time


class FrameGrabber:
    """Wraps cv2.VideoCapture with frame skipping.

    Every frame is returned for display, but only every Nth frame
    has should_process=True for running detection/tracking.
    """

    def __init__(self, cap, frame_stride=3):
        self._cap = cap
        self.frame_stride = frame_stride
        self._frame_idx = 0
        self.fps = cap.get(5) or 30.0  # CAP_PROP_FPS
        self.width = int(cap.get(3))   # CAP_PROP_FRAME_WIDTH
        self.height = int(cap.get(4))  # CAP_PROP_FRAME_HEIGHT

    def next(self):
        """Returns (frame, should_process, frame_idx) or (None, False, -1)."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            return None, False, -1
        idx = self._frame_idx
        should_process = (idx % self.frame_stride) == 0
        self._frame_idx += 1
        return frame, should_process, idx

    def raise_stride(self):
        """Increase stride for adaptive performance. Returns new stride."""
        steps = [3, 5, 8, 12]
        for s in steps:
            if s > self.frame_stride:
                self.frame_stride = s
                return s
        return self.frame_stride

    def release(self):
        self._cap.release()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_capture.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/capture.py tests/test_capture.py
git commit -m "feat: camera capture with frame skipping and adaptive stride"
```

---

### Task 3: Vehicle + Plate Detector

**Files:**
- Create: `src/detector.py`

This module wraps ultralytics YOLO and returns normalized detection dicts. No unit tests for this task — it requires real YOLO models. Integration tested in Task 11.

- [ ] **Step 1: Implement detector.py**

```python
# src/detector.py
from ultralytics import YOLO

# COCO class IDs for vehicles
_VEHICLE_COCO_IDS = {2: "car", 5: "bus", 7: "truck"}


class Detector:
    """Runs two YOLO models (vehicle + plate) on a frame."""

    def __init__(self, vehicle_model_path, plate_model_path,
                 vehicle_conf=0.3, plate_conf=0.3, imgsz=960):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.imgsz = imgsz

    def detect(self, frame):
        """Run both detectors on frame. Returns list of detection dicts.

        Each dict has: bbox_xyxy, conf, class_name, object_type
        bbox_xyxy is in pixel coordinates of the input frame.
        """
        detections = []

        # Vehicle detection
        veh_results = self.vehicle_model(
            frame, conf=self.vehicle_conf, imgsz=self.imgsz, verbose=False
        )
        for r in veh_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in _VEHICLE_COCO_IDS:
                    detections.append({
                        "bbox_xyxy": box.xyxy[0].cpu().numpy().tolist(),
                        "conf": float(box.conf[0]),
                        "class_name": _VEHICLE_COCO_IDS[cls_id],
                        "object_type": "vehicle",
                    })

        # Plate detection
        plt_results = self.plate_model(
            frame, conf=self.plate_conf, imgsz=self.imgsz, verbose=False
        )
        for r in plt_results:
            for box in r.boxes:
                detections.append({
                    "bbox_xyxy": box.xyxy[0].cpu().numpy().tolist(),
                    "conf": float(box.conf[0]),
                    "class_name": "license_plate",
                    "object_type": "plate",
                })

        return detections
```

- [ ] **Step 2: Commit**

```bash
git add src/detector.py
git commit -m "feat: dual YOLO detector wrapper for vehicles and plates"
```

---

### Task 4: Multi-Object Tracker

**Files:**
- Create: `src/tracker.py`
- Create: `tests/test_tracker.py`

- [ ] **Step 1: Write tracker tests**

```python
# tests/test_tracker.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_tracker.py -v`
Expected: FAIL

- [ ] **Step 3: Implement tracker.py**

```python
# src/tracker.py
import time
import numpy as np


def _iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _centroid_dist(box_a, box_b):
    ca, cb = _centroid(box_a), _centroid(box_b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def _size_ratio(box_a, box_b):
    a, b = _box_area(box_a), _box_area(box_b)
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def _match_score(det_box, track_box):
    """Combined matching score: higher = better match."""
    iou = _iou(det_box, track_box)
    size_sim = _size_ratio(det_box, track_box)
    # Centroid distance penalty (normalized by diagonal of track box)
    diag = ((track_box[2] - track_box[0]) ** 2 + (track_box[3] - track_box[1]) ** 2) ** 0.5
    if diag == 0:
        return 0.0
    dist_norm = _centroid_dist(det_box, track_box) / diag
    dist_score = max(0, 1.0 - dist_norm)
    return 0.4 * iou + 0.3 * size_sim + 0.3 * dist_score


class Tracker:
    """Simple multi-object tracker using centroid + IoU + size matching."""

    MATCH_THRESHOLD = 0.25

    def __init__(self, track_timeout=4.0):
        self.track_timeout = track_timeout
        self.active_tracks = {}  # track_id -> track dict
        self._next_veh_id = 0
        self._next_plt_id = 0

    def _new_id(self, object_type):
        if object_type == "vehicle":
            tid = f"veh_{self._next_veh_id}"
            self._next_veh_id += 1
        else:
            tid = f"plt_{self._next_plt_id}"
            self._next_plt_id += 1
        return tid

    def update(self, detections):
        """Update tracks with new detections. Returns list of active track dicts.

        Each returned dict has: track_id, object_type, bbox_xyxy, conf, class_name,
        first_seen_ts, last_seen_ts, last_saved_ts, save_count, best_quality_score,
        best_crop_path, is_new
        """
        now = time.time()

        # Remove stale tracks
        stale = [tid for tid, t in self.active_tracks.items()
                 if now - t["last_seen_ts"] > self.track_timeout]
        for tid in stale:
            del self.active_tracks[tid]

        # Split detections by type for class-consistent matching
        matched_track_ids = set()
        matched_det_indices = set()

        # Build list of (det_idx, det) pairs
        det_list = list(enumerate(detections))

        # Greedy matching: score all pairs, pick best first
        pairs = []
        for di, det in det_list:
            for tid, trk in self.active_tracks.items():
                if det["object_type"] != trk["object_type"]:
                    continue
                score = _match_score(det["bbox_xyxy"], trk["bbox_xyxy"])
                if score >= self.MATCH_THRESHOLD:
                    pairs.append((score, di, tid))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for score, di, tid in pairs:
            if di in matched_det_indices or tid in matched_track_ids:
                continue
            # Update existing track
            det = detections[di]
            trk = self.active_tracks[tid]
            trk["bbox_xyxy"] = det["bbox_xyxy"]
            trk["conf"] = det["conf"]
            trk["class_name"] = det["class_name"]
            trk["last_seen_ts"] = now
            trk["is_new"] = False
            matched_track_ids.add(tid)
            matched_det_indices.add(di)

        # Create new tracks for unmatched detections
        for di, det in det_list:
            if di in matched_det_indices:
                continue
            tid = self._new_id(det["object_type"])
            self.active_tracks[tid] = {
                "track_id": tid,
                "object_type": det["object_type"],
                "bbox_xyxy": det["bbox_xyxy"],
                "conf": det["conf"],
                "class_name": det["class_name"],
                "first_seen_ts": now,
                "last_seen_ts": now,
                "last_saved_ts": None,
                "save_count": 0,
                "best_quality_score": 0.0,
                "best_crop_path": None,
                "is_new": True,
            }

        return list(self.active_tracks.values())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_tracker.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/tracker.py tests/test_tracker.py
git commit -m "feat: multi-object tracker with centroid+IoU+size matching"
```

---

### Task 5: Vehicle-Plate Association

**Files:**
- Create: `src/association.py`
- Create: `tests/test_association.py`

- [ ] **Step 1: Write association tests**

```python
# tests/test_association.py
from src.association import associate_plates


def _trk(track_id, x1, y1, x2, y2, obj_type):
    return {
        "track_id": track_id,
        "object_type": obj_type,
        "bbox_xyxy": [x1, y1, x2, y2],
        "conf": 0.8,
        "class_name": "car" if obj_type == "vehicle" else "license_plate",
    }


def test_plate_inside_vehicle_matched():
    vehicles = [_trk("veh_0", 100, 100, 500, 400, "vehicle")]
    plates = [_trk("plt_0", 200, 350, 350, 390, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "matched"
    assert result["plt_0"]["vehicle_track_id"] == "veh_0"


def test_plate_no_vehicle_unmatched():
    vehicles = []
    plates = [_trk("plt_0", 200, 350, 350, 390, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "unmatched"


def test_plate_ambiguous_multiple_vehicles():
    # Plate overlaps two vehicles equally
    vehicles = [
        _trk("veh_0", 100, 100, 350, 400, "vehicle"),
        _trk("veh_1", 300, 100, 550, 400, "vehicle"),
    ]
    plates = [_trk("plt_0", 310, 350, 340, 390, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] in ("matched", "ambiguous")


def test_plate_below_vehicle_provisional():
    # Plate is below vehicle box (truck bumper scenario)
    vehicles = [_trk("veh_0", 100, 100, 500, 300, "vehicle")]
    plates = [_trk("plt_0", 200, 310, 350, 340, "plate")]  # below by 10px
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "matched_provisional"
    assert result["plt_0"]["vehicle_track_id"] == "veh_0"


def test_plate_too_far_below_unmatched():
    vehicles = [_trk("veh_0", 100, 100, 500, 300, "vehicle")]
    plates = [_trk("plt_0", 200, 400, 350, 430, "plate")]  # 100px below
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "unmatched"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_association.py -v`
Expected: FAIL

- [ ] **Step 3: Implement association.py**

```python
# src/association.py


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _overlap_ratio(plate_box, vehicle_box):
    """Fraction of plate area that overlaps with vehicle box."""
    x1 = max(plate_box[0], vehicle_box[0])
    y1 = max(plate_box[1], vehicle_box[1])
    x2 = min(plate_box[2], vehicle_box[2])
    y2 = min(plate_box[3], vehicle_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    plate_area = (plate_box[2] - plate_box[0]) * (plate_box[3] - plate_box[1])
    return inter / plate_area if plate_area > 0 else 0.0


def _x_overlap_ratio(plate_box, vehicle_box):
    """Fraction of plate x-range that overlaps vehicle x-range."""
    overlap = min(plate_box[2], vehicle_box[2]) - max(plate_box[0], vehicle_box[0])
    plate_w = plate_box[2] - plate_box[0]
    return max(0, overlap) / plate_w if plate_w > 0 else 0.0


def _score_match(plate_box, vehicle_box):
    """Score how well a plate belongs to a vehicle. Higher = better."""
    pcx, pcy = _center(plate_box)
    vx1, vy1, vx2, vy2 = vehicle_box
    vw = vx2 - vx1
    vh = vy2 - vy1

    # Containment: is plate center inside vehicle?
    inside = vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2
    containment = 1.0 if inside else 0.0

    # Overlap ratio
    overlap = _overlap_ratio(plate_box, vehicle_box)

    # Lower-half preference: plates are usually near bottom of vehicle
    if vh > 0 and inside:
        y_frac = (pcy - vy1) / vh  # 0=top, 1=bottom
        lower_bonus = y_frac  # higher score when plate is lower
    else:
        lower_bonus = 0.0

    # Size plausibility: plate should be much smaller than vehicle
    plate_area = (plate_box[2] - plate_box[0]) * (plate_box[3] - plate_box[1])
    vehicle_area = vw * vh
    if vehicle_area > 0:
        size_ratio = plate_area / vehicle_area
        size_ok = 1.0 if size_ratio < 0.15 else max(0, 1.0 - (size_ratio - 0.15) * 5)
    else:
        size_ok = 0.0

    return 0.3 * containment + 0.25 * overlap + 0.25 * lower_bonus + 0.2 * size_ok


def associate_plates(vehicles, plates, proximity_margin_px=60):
    """Associate plates to vehicles using scored matching.

    Returns dict: plate_track_id -> {status, vehicle_track_id, score}
    status is one of: matched, matched_provisional, ambiguous, unmatched
    """
    result = {}

    for plate in plates:
        pbox = plate["bbox_xyxy"]
        pid = plate["track_id"]

        candidates = []
        for veh in vehicles:
            vbox = veh["bbox_xyxy"]
            score = _score_match(pbox, vbox)
            if score > 0.15:
                candidates.append((score, veh["track_id"], "matched"))

        # Proximity check for provisional matches (trucks/buses)
        if not candidates:
            ptop = pbox[1]
            for veh in vehicles:
                vbox = veh["bbox_xyxy"]
                vbottom = vbox[3]
                gap = ptop - vbottom
                if 0 <= gap <= proximity_margin_px:
                    x_overlap = _x_overlap_ratio(pbox, vbox)
                    if x_overlap >= 0.5:
                        candidates.append((0.1, veh["track_id"], "matched_provisional"))

        if not candidates:
            result[pid] = {"status": "unmatched", "vehicle_track_id": None, "score": 0.0}
        elif len(candidates) == 1:
            score, vid, status = candidates[0]
            result[pid] = {"status": status, "vehicle_track_id": vid, "score": score}
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score = candidates[0][0]
            second_score = candidates[1][0]
            if best_score > second_score * 1.5:
                score, vid, status = candidates[0]
                result[pid] = {"status": status, "vehicle_track_id": vid, "score": score}
            else:
                result[pid] = {"status": "ambiguous", "vehicle_track_id": None, "score": best_score}

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_association.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/association.py tests/test_association.py
git commit -m "feat: scored vehicle-plate association with proximity matching"
```

---

### Task 6: Quality Gate

**Files:**
- Create: `src/quality.py`
- Create: `tests/test_quality.py`

- [ ] **Step 1: Write quality gate tests**

```python
# tests/test_quality.py
import numpy as np
from src.quality import check_quality


def test_good_vehicle_passes():
    crop = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
    result = check_quality(crop, "vehicle", bbox_xyxy=[100, 100, 400, 300],
                           image_shape=(720, 1280),
                           min_plate_w=40, min_plate_h=14,
                           min_vehicle_area=12000, min_blur_score=10.0)
    assert result["passes"] is True


def test_small_plate_rejected():
    crop = np.random.randint(50, 200, (10, 20, 3), dtype=np.uint8)
    result = check_quality(crop, "plate", bbox_xyxy=[100, 100, 120, 110],
                           image_shape=(720, 1280),
                           min_plate_w=40, min_plate_h=14,
                           min_vehicle_area=12000, min_blur_score=10.0)
    assert result["passes"] is False
    assert "size" in result["reject_reason"]


def test_small_vehicle_rejected():
    crop = np.random.randint(50, 200, (50, 50, 3), dtype=np.uint8)
    result = check_quality(crop, "vehicle", bbox_xyxy=[100, 100, 150, 150],
                           image_shape=(720, 1280),
                           min_plate_w=40, min_plate_h=14,
                           min_vehicle_area=12000, min_blur_score=10.0)
    assert result["passes"] is False
    assert "size" in result["reject_reason"]


def test_truncated_crop_rejected():
    crop = np.random.randint(50, 200, (100, 200, 3), dtype=np.uint8)
    # bbox extends beyond image boundary by >30%
    result = check_quality(crop, "vehicle", bbox_xyxy=[-100, 100, 200, 300],
                           image_shape=(720, 1280),
                           min_plate_w=40, min_plate_h=14,
                           min_vehicle_area=12000, min_blur_score=10.0)
    assert result["passes"] is False
    assert "truncat" in result["reject_reason"]


def test_quality_score_returned():
    crop = np.random.randint(50, 200, (200, 300, 3), dtype=np.uint8)
    result = check_quality(crop, "vehicle", bbox_xyxy=[100, 100, 400, 300],
                           image_shape=(720, 1280),
                           min_plate_w=40, min_plate_h=14,
                           min_vehicle_area=12000, min_blur_score=10.0)
    assert 0.0 <= result["quality_score"] <= 1.0
    assert result["blur_score"] > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_quality.py -v`
Expected: FAIL

- [ ] **Step 3: Implement quality.py**

```python
# src/quality.py
import cv2
import numpy as np


def _blur_score(crop):
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def _truncation_fraction(bbox_xyxy, image_shape):
    """Fraction of bbox area that falls outside the image."""
    h, w = image_shape[:2]
    x1, y1, x2, y2 = bbox_xyxy
    # Clipped box
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
    """Basic exposure sanity: not too dark, not too bright."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    mean_val = gray.mean()
    return 20 < mean_val < 240


def check_quality(crop, object_type, bbox_xyxy, image_shape,
                  min_plate_w=40, min_plate_h=14,
                  min_vehicle_area=12000, min_blur_score=50.0):
    """Check if a crop is worth saving.

    Returns dict with: passes, quality_score, blur_score, reject_reason, truncated
    """
    h, w = crop.shape[:2]
    blur = _blur_score(crop)
    trunc_frac = _truncation_fraction(bbox_xyxy, image_shape)
    truncated = trunc_frac > 0.05
    exposure = _exposure_ok(crop)

    # Size checks
    if object_type == "plate":
        if w < min_plate_w or h < min_plate_h:
            return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                    "reject_reason": "size_too_small", "truncated": truncated}
    else:  # vehicle
        if w * h < min_vehicle_area:
            return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                    "reject_reason": "size_too_small", "truncated": truncated}

    # Truncation check (>30% outside frame = reject)
    if trunc_frac > 0.30:
        return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                "reject_reason": "truncated", "truncated": True}

    # Blur check
    if blur < min_blur_score:
        return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                "reject_reason": "too_blurry", "truncated": truncated}

    # Exposure check
    if not exposure:
        return {"passes": False, "quality_score": 0.0, "blur_score": blur,
                "reject_reason": "bad_exposure", "truncated": truncated}

    # Compute quality score (0-1)
    blur_norm = min(blur / 500.0, 1.0)
    size_norm = min((w * h) / 100000.0, 1.0) if object_type == "vehicle" else min(w / 200.0, 1.0)
    trunc_penalty = 1.0 - trunc_frac
    quality_score = 0.4 * blur_norm + 0.35 * size_norm + 0.25 * trunc_penalty

    return {"passes": True, "quality_score": quality_score, "blur_score": blur,
            "reject_reason": None, "truncated": truncated}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_quality.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/quality.py tests/test_quality.py
git commit -m "feat: quality gate with blur, size, truncation, and exposure checks"
```

---

### Task 7: Save Decision Engine + File I/O

**Files:**
- Create: `src/saver.py`
- Create: `tests/test_saver.py`

- [ ] **Step 1: Write saver tests**

```python
# tests/test_saver.py
import time
import numpy as np
import json
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
    # Much better quality should override cooldown
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_saver.py -v`
Expected: FAIL

- [ ] **Step 3: Implement saver.py**

```python
# src/saver.py
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

_CLASS_MAP = {"vehicle": 0, "plate": 1}


class SaveDecider:
    """Decides whether a track should be saved based on cooldown and quality."""

    BETTER_THRESHOLD = 0.2  # quality must improve by this much to override cooldown

    def __init__(self, vehicle_cooldown=3.0, plate_cooldown=2.0, save_better_only=True):
        self.cooldowns = {"vehicle": vehicle_cooldown, "plate": plate_cooldown}
        self.save_better_only = save_better_only

    def should_save(self, track, quality_score):
        """Returns True if this track should be saved now."""
        # New track always saves
        if track["is_new"] or track["save_count"] == 0:
            return True

        cooldown = self.cooldowns.get(track["object_type"], 3.0)
        now = time.time()
        elapsed = now - (track["last_saved_ts"] or 0)

        # Better quality overrides cooldown
        if self.save_better_only and quality_score > track["best_quality_score"] + self.BETTER_THRESHOLD:
            return True

        # Cooldown expired
        if elapsed >= cooldown:
            return True

        return False


def _date_subdir():
    return datetime.now().strftime("%Y-%m-%d")


def save_frame(frame, output_dir, session_id, timestamp_ms, frame_idx):
    """Save full frame as JPEG 95%. Returns Path to saved file."""
    subdir = Path(output_dir) / "raw" / "frames" / _date_subdir()
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"frm_{session_id}_{timestamp_ms}_{frame_idx:06d}.jpg"
    path = subdir / fname
    cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path


def save_crop(crop, output_dir, session_id, track_id, timestamp_ms, object_type):
    """Save crop as PNG with padding already applied. Returns Path."""
    kind = "vehicles" if object_type == "vehicle" else "plates"
    prefix = "veh" if object_type == "vehicle" else "plt"
    subdir = Path(output_dir) / "raw" / "crops" / kind / _date_subdir()
    subdir.mkdir(parents=True, exist_ok=True)
    fname = f"{prefix}_{session_id}_{track_id}_{timestamp_ms}.png"
    path = subdir / fname
    cv2.imwrite(str(path), crop)
    return path


def save_label(detections, img_w, img_h, output_dir, session_id, timestamp_ms, frame_idx):
    """Save pseudo-labels in YOLO format. Returns Path."""
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
    """Extract a padded crop from frame. Returns cropped numpy array."""
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_saver.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/saver.py tests/test_saver.py
git commit -m "feat: save decision engine with cooldown, quality override, and file I/O"
```

---

### Task 8: Metadata Logger

**Files:**
- Create: `src/metadata.py`
- Create: `tests/test_metadata.py`

- [ ] **Step 1: Write metadata tests**

```python
# tests/test_metadata.py
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_metadata.py -v`
Expected: FAIL

- [ ] **Step 3: Implement metadata.py**

```python
# src/metadata.py
import json
from pathlib import Path


class MetadataLogger:
    """Writes JSONL metadata files for save events, tracks, and sessions."""

    def __init__(self, output_dir):
        self._output_dir = Path(output_dir)
        self._meta_dir = self._output_dir / "raw" / "metadata"
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        self._event_counter = 0

    def next_event_id(self):
        self._event_counter += 1
        return f"evt_{self._event_counter:06d}"

    def _append_jsonl(self, filename, data):
        path = self._meta_dir / filename
        with open(path, "a") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def log_save_event(self, event):
        self._append_jsonl("save_events.jsonl", event)

    def log_track(self, track_data):
        self._append_jsonl("tracks.jsonl", track_data)

    def log_session(self, session_data):
        self._append_jsonl("sessions.jsonl", session_data)

    def ensure_classes_txt(self):
        classes_path = self._output_dir / "raw" / "classes.txt"
        classes_path.parent.mkdir(parents=True, exist_ok=True)
        classes_path.write_text("vehicle\nplate\n")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_metadata.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/metadata.py tests/test_metadata.py
git commit -m "feat: JSONL metadata logger for events, tracks, and sessions"
```

---

### Task 9: Live Display

**Files:**
- Create: `src/display.py`

No unit tests — OpenCV GUI is tested visually in integration. The module is pure rendering logic with no side effects beyond drawing on the frame.

- [ ] **Step 1: Implement display.py**

```python
# src/display.py
import cv2


class Display:
    """OpenCV live preview with bounding boxes, association lines, and counters."""

    VEHICLE_COLOR = (0, 255, 0)   # Green
    PLATE_COLOR = (0, 0, 255)     # Red
    LINE_COLOR = (255, 200, 0)    # Cyan-ish
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.window_name = "Syrian Plate Collector"

    def draw(self, frame, tracks, associations, stats):
        """Draw boxes, labels, association lines, and stats overlay.

        Args:
            frame: BGR numpy array (will be modified in-place)
            tracks: list of track dicts with bbox_xyxy, object_type, track_id
            associations: dict from associate_plates() — plate_track_id -> {status, vehicle_track_id}
            stats: dict with keys: vehicles, plates, active_tracks, saved, rejected
        Returns:
            frame with drawings
        """
        vis = frame.copy()

        # Index tracks by id for line drawing
        track_by_id = {t["track_id"]: t for t in tracks}

        for trk in tracks:
            x1, y1, x2, y2 = [int(v) for v in trk["bbox_xyxy"]]
            if trk["object_type"] == "vehicle":
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.VEHICLE_COLOR, 2)
                label = f"VEHICLE {trk['track_id']}"
                cv2.putText(vis, label, (x1, y1 - 5), self.FONT, 0.5, self.VEHICLE_COLOR, 1)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.PLATE_COLOR, 2)
                label = f"PLATE {trk['track_id']}"
                cv2.putText(vis, label, (x1, y1 - 5), self.FONT, 0.5, self.PLATE_COLOR, 1)

        # Draw association lines
        for pid, assoc in associations.items():
            if assoc["vehicle_track_id"] and pid in track_by_id and assoc["vehicle_track_id"] in track_by_id:
                pbox = track_by_id[pid]["bbox_xyxy"]
                vbox = track_by_id[assoc["vehicle_track_id"]]["bbox_xyxy"]
                pcx = int((pbox[0] + pbox[2]) / 2)
                pcy = int((pbox[1] + pbox[3]) / 2)
                vcx = int((vbox[0] + vbox[2]) / 2)
                vcy = int((vbox[1] + vbox[3]) / 2)
                cv2.line(vis, (pcx, pcy), (vcx, vcy), self.LINE_COLOR, 1)

        # Stats overlay
        y = 25
        for key, val in stats.items():
            text = f"{key}: {val}"
            cv2.putText(vis, text, (10, y), self.FONT, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, text, (10, y), self.FONT, 0.6, (0, 0, 0), 1)
            y += 25

        return vis

    def show(self, frame):
        """Display frame and return True if user pressed 'q'."""
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def close(self):
        cv2.destroyAllWindows()
```

- [ ] **Step 2: Commit**

```bash
git add src/display.py
git commit -m "feat: OpenCV live display with boxes, association lines, and stats"
```

---

### Task 10: Main Pipeline Orchestrator

**Files:**
- Create: `src/collector.py`

This is the main entry point that wires all modules together. Tested via manual integration (requires camera + models).

- [ ] **Step 1: Implement collector.py**

```python
# src/collector.py
import time
import uuid
from datetime import datetime, timezone

import cv2

from src.config import parse_args
from src.capture import FrameGrabber
from src.detector import Detector
from src.tracker import Tracker
from src.association import associate_plates
from src.quality import check_quality
from src.saver import SaveDecider, save_frame, save_crop, save_label, extract_crop
from src.metadata import MetadataLogger
from src.display import Display


def _generate_session_id(camera_id):
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:4]
    return f"sess_{date_str}_{camera_id}_{short_uuid}"


def run(argv=None):
    args = parse_args(argv)

    # Session setup
    camera_id = f"cam{args.camera}"
    session_id = args.session_id or _generate_session_id(camera_id)
    print(f"[collector] Session: {session_id}")
    print(f"[collector] Output: {args.output}")

    # Initialize modules
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    grabber = FrameGrabber(cap, frame_stride=args.frame_stride)
    print(f"[collector] Camera {args.camera}: {grabber.width}x{grabber.height} @ {grabber.fps:.0f}fps")
    print(f"[collector] Frame stride: {args.frame_stride} (~{grabber.fps / args.frame_stride:.0f} inference FPS)")

    detector = Detector(
        args.vehicle_model, args.plate_model,
        vehicle_conf=args.vehicle_conf, plate_conf=args.plate_conf,
        imgsz=args.imgsz,
    )
    tracker = Tracker(track_timeout=args.track_timeout)
    save_decider = SaveDecider(
        vehicle_cooldown=args.vehicle_cooldown,
        plate_cooldown=args.plate_cooldown,
        save_better_only=args.save_better_only,
    )
    meta = MetadataLogger(args.output)
    meta.ensure_classes_txt()
    display = Display() if not args.no_display else None

    # Session stats
    stats = {
        "vehicles": 0, "plates": 0, "active_tracks": 0,
        "saved": 0, "rejected": 0,
    }
    session_start = datetime.now(timezone.utc)
    total_frames = 0
    last_tracks = []
    last_associations = {}
    last_inference_time = 0

    print("[collector] Running... Press 'q' to quit.")

    try:
        while True:
            frame, should_process, frame_idx = grabber.next()
            if frame is None:
                break
            total_frames += 1

            if should_process:
                t0 = time.time()

                # Detect
                detections = detector.detect(frame)

                # Track
                tracks = tracker.update(detections)
                last_tracks = tracks

                # Split tracks by type
                vehicle_tracks = [t for t in tracks if t["object_type"] == "vehicle"]
                plate_tracks = [t for t in tracks if t["object_type"] == "plate"]

                # Associate
                associations = associate_plates(
                    vehicle_tracks, plate_tracks,
                    proximity_margin_px=args.proximity_margin_px,
                )
                last_associations = associations

                # Update stats
                stats["vehicles"] = len(vehicle_tracks)
                stats["plates"] = len(plate_tracks)
                stats["active_tracks"] = len(tracks)

                # Quality check + save decision for each track
                timestamp_ms = int(time.time() * 1000)
                save_worthy_dets = []

                for trk in tracks:
                    bbox = trk["bbox_xyxy"]
                    padding = 0.10 if trk["object_type"] == "vehicle" else 0.05
                    crop = extract_crop(frame, bbox, padding_frac=padding)

                    if crop.size == 0:
                        continue

                    qr = check_quality(
                        crop, trk["object_type"], bbox, frame.shape,
                        min_plate_w=args.min_plate_w,
                        min_plate_h=args.min_plate_h,
                        min_vehicle_area=args.min_vehicle_area,
                        min_blur_score=args.min_blur_score,
                    )

                    if not qr["passes"]:
                        stats["rejected"] += 1
                        continue

                    if not save_decider.should_save(trk, qr["quality_score"]):
                        continue

                    # Save crop
                    crop_path = save_crop(
                        crop, args.output, session_id,
                        trk["track_id"], timestamp_ms, trk["object_type"],
                    )

                    # Update track state
                    trk["last_saved_ts"] = time.time()
                    trk["save_count"] += 1
                    if qr["quality_score"] > trk["best_quality_score"]:
                        trk["best_quality_score"] = qr["quality_score"]
                        trk["best_crop_path"] = str(crop_path)

                    # Build detection record for metadata
                    det_record = {
                        "object_type": trk["object_type"],
                        "track_id": trk["track_id"],
                        "vehicle_type": trk["class_name"] if trk["object_type"] == "vehicle" else None,
                        "bbox_xyxy": [int(v) for v in bbox],
                        "detector_conf": trk["conf"],
                        "crop_path": str(crop_path),
                        "quality_score": round(qr["quality_score"], 3),
                        "blur_score": round(qr["blur_score"], 1),
                        "truncated": qr["truncated"],
                        "occluded": False,
                        "review_status": "pending",
                        "negative_type": None,
                    }

                    # Add association info
                    if trk["object_type"] == "plate" and trk["track_id"] in associations:
                        assoc = associations[trk["track_id"]]
                        det_record["association_status"] = assoc["status"]
                        det_record["associated_vehicle_track_id"] = assoc["vehicle_track_id"]
                        det_record["association_score"] = round(assoc["score"], 3)
                    elif trk["object_type"] == "vehicle":
                        # Find associated plate
                        linked_plate = None
                        for pid, a in associations.items():
                            if a["vehicle_track_id"] == trk["track_id"]:
                                linked_plate = pid
                                break
                        det_record["associated_plate_track_id"] = linked_plate

                    save_worthy_dets.append(det_record)

                # Save frame + labels if anything was saved
                if save_worthy_dets:
                    frame_path = save_frame(frame, args.output, session_id, timestamp_ms, frame_idx)
                    label_path = save_label(
                        save_worthy_dets, frame.shape[1], frame.shape[0],
                        args.output, session_id, timestamp_ms, frame_idx,
                    )

                    # Log metadata
                    event = {
                        "event_id": meta.next_event_id(),
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "camera_id": camera_id,
                        "frame_path": str(frame_path),
                        "label_path": str(label_path),
                        "image_width": frame.shape[1],
                        "image_height": frame.shape[0],
                        "detections": save_worthy_dets,
                    }
                    meta.log_save_event(event)
                    stats["saved"] += len(save_worthy_dets)

                # Adaptive stride
                last_inference_time = time.time() - t0
                frame_budget = 1.0 / (grabber.fps / grabber.frame_stride)
                if args.adaptive_stride and last_inference_time > frame_budget * 1.5:
                    new_stride = grabber.raise_stride()
                    print(f"[collector] Adaptive stride raised to {new_stride}")

            # Display (show every frame for smooth preview)
            if display:
                vis = display.draw(frame, last_tracks, last_associations, stats)
                if display.show(vis):
                    break

    except KeyboardInterrupt:
        print("\n[collector] Interrupted.")
    finally:
        # Log session summary
        session_end = datetime.now(timezone.utc)
        meta.log_session({
            "session_id": session_id,
            "camera_id": camera_id,
            "start_time": session_start.isoformat(),
            "end_time": session_end.isoformat(),
            "duration_seconds": (session_end - session_start).total_seconds(),
            "total_frames": total_frames,
            "total_saved": stats["saved"],
            "total_rejected": stats["rejected"],
        })

        # Log final track states
        for trk in tracker.active_tracks.values():
            meta.log_track({
                "track_id": trk["track_id"],
                "object_type": trk["object_type"],
                "first_seen_ts": trk["first_seen_ts"],
                "last_seen_ts": trk["last_seen_ts"],
                "save_count": trk["save_count"],
                "best_quality_score": trk["best_quality_score"],
                "best_crop_path": trk["best_crop_path"],
            })

        grabber.release()
        if display:
            display.close()
        print(f"[collector] Done. {stats['saved']} saves, {stats['rejected']} rejected, {total_frames} frames.")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2: Commit**

```bash
git add src/collector.py
git commit -m "feat: main collector pipeline orchestrating all modules"
```

---

### Task 11: Pre-flight Validation Script

**Files:**
- Create: `preflight.py`

- [ ] **Step 1: Implement preflight.py**

```python
# preflight.py
"""Pre-flight validation: test plate detector recall on sample images."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def run_preflight(plate_model_path, test_dir, conf=0.3, imgsz=960):
    model = YOLO(plate_model_path)
    test_dir = Path(test_dir)
    images = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))

    if not images:
        print(f"[preflight] No images found in {test_dir}")
        return

    total = len(images)
    detected = 0
    all_confs = []

    print(f"[preflight] Testing {total} images with model: {plate_model_path}")
    print(f"[preflight] Confidence threshold: {conf}")
    print("-" * 60)

    for img_path in images:
        results = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)
        n_plates = 0
        for r in results:
            n_plates += len(r.boxes)
            for box in r.boxes:
                all_confs.append(float(box.conf[0]))

        status = "OK" if n_plates > 0 else "MISS"
        if n_plates > 0:
            detected += 1
        print(f"  [{status}] {img_path.name}: {n_plates} plate(s)")

    recall = detected / total * 100
    print("-" * 60)
    print(f"[preflight] Recall: {detected}/{total} = {recall:.0f}%")
    if all_confs:
        avg_conf = sum(all_confs) / len(all_confs)
        print(f"[preflight] Avg confidence: {avg_conf:.2f}")

    if recall >= 70:
        print("[preflight] PASS — proceed with this model.")
    elif recall >= 50:
        print("[preflight] WARN — proceed but flag; consider fine-tuning sooner.")
    else:
        print("[preflight] FAIL — swap to fallback model before collection.")

    return recall


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-flight plate detector validation")
    p.add_argument("--plate-model", required=True, help="Plate model path or HF repo")
    p.add_argument("--test-dir", required=True, help="Directory of test plate images")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()
    run_preflight(args.plate_model, args.test_dir, args.conf, args.imgsz)
```

- [ ] **Step 2: Commit**

```bash
git add preflight.py
git commit -m "feat: pre-flight plate model validation script"
```

---

## Phase 1.5 — AI Review Pipeline

### Task 12: Grid Compositor for Batched API Calls

**Files:**
- Create: `src/review/__init__.py`
- Create: `src/review/grid.py`
- Create: `tests/test_review.py`

- [ ] **Step 1: Write grid compositor tests**

```python
# tests/test_review.py
import numpy as np
from pathlib import Path
from src.review.grid import compose_grid, parse_grid_size


def test_parse_grid_size():
    assert parse_grid_size("3x3") == (3, 3)
    assert parse_grid_size("2x2") == (2, 2)
    assert parse_grid_size("4x4") == (4, 4)


def test_compose_grid_creates_image():
    # Create 4 fake images
    images = [np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8) for _ in range(4)]
    paths = [Path(f"img_{i}.png") for i in range(4)]
    composite, mapping = compose_grid(images, paths, grid_size=(2, 2), tile_px=256)
    # 2x2 grid of 256px tiles = 512x512
    assert composite.shape == (512, 512, 3)
    assert len(mapping) == 4
    assert mapping[0] == paths[0]


def test_compose_grid_partial_batch():
    # Only 3 images for a 2x2 grid — last cell should be black
    images = [np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8) for _ in range(3)]
    paths = [Path(f"img_{i}.png") for i in range(3)]
    composite, mapping = compose_grid(images, paths, grid_size=(2, 2), tile_px=256)
    assert composite.shape == (512, 512, 3)
    assert len(mapping) == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_review.py -v`
Expected: FAIL

- [ ] **Step 3: Implement grid.py**

```python
# src/review/__init__.py
# (empty)

# src/review/grid.py
import cv2
import numpy as np
from pathlib import Path


def parse_grid_size(s):
    """Parse '3x3' into (3, 3)."""
    parts = s.lower().split("x")
    return int(parts[0]), int(parts[1])


def _resize_with_padding(img, tile_px):
    """Resize image to fit in tile_px x tile_px, preserving aspect ratio, with black padding."""
    h, w = img.shape[:2]
    scale = min(tile_px / w, tile_px / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((tile_px, tile_px, 3), dtype=np.uint8)
    y_off = (tile_px - new_h) // 2
    x_off = (tile_px - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def compose_grid(images, paths, grid_size=(3, 3), tile_px=512):
    """Compose a grid of images into a single composite image.

    Args:
        images: list of BGR numpy arrays
        paths: list of Path objects corresponding to each image
        grid_size: (rows, cols) tuple
        tile_px: size of each cell in pixels

    Returns:
        (composite_image, index_to_path_mapping)
        mapping is dict: cell_index -> source_path
    """
    rows, cols = grid_size
    canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
    mapping = {}

    for i, (img, path) in enumerate(zip(images, paths)):
        if i >= rows * cols:
            break
        r, c = divmod(i, cols)
        tile = _resize_with_padding(img, tile_px)

        # Draw cell index label
        cv2.putText(tile, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        y1, y2 = r * tile_px, (r + 1) * tile_px
        x1, x2 = c * tile_px, (c + 1) * tile_px
        canvas[y1:y2, x1:x2] = tile
        mapping[i] = path

    return canvas, mapping
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_review.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/review/__init__.py src/review/grid.py tests/test_review.py
git commit -m "feat: grid compositor for batched AI review API calls"
```

---

### Task 13: AI Review Modules (bbox, brand, plate)

**Files:**
- Create: `src/review/bbox.py`
- Create: `src/review/brand.py`
- Create: `src/review/plate.py`

- [ ] **Step 1: Implement bbox verification module**

```python
# src/review/bbox.py
"""Module 1: Verify bounding box accuracy using Vision API."""
import json
import base64
from pathlib import Path

import cv2

from src.review.grid import compose_grid, parse_grid_size


def _encode_image(img):
    """Encode BGR numpy array to base64 JPEG."""
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode("utf-8")


def _build_prompt(n_cells):
    return f"""You are reviewing vehicle and license plate detection bounding boxes.

This image is a grid of {n_cells} frames. Each cell has an index number (0-{n_cells-1}) in the top-left.
The original frames had bounding boxes drawn on them before being placed in the grid.

For each cell, evaluate the bounding box quality and return a JSON array of length {n_cells} in row-major order.

Each element must have:
- "cell_index": integer
- "bbox_quality": "good" | "acceptable" | "poor"
- "missing_objects": list of descriptions of vehicles/plates visible but not boxed
- "false_positives": list of descriptions of boxes around non-vehicle/non-plate objects
- "certainty": "high" | "medium" | "low"

If a cell is black/empty, return {{"cell_index": N, "bbox_quality": "good", "missing_objects": [], "false_positives": [], "certainty": "low"}}

Return ONLY the JSON array, no other text."""


def verify_bbox_batch(frame_paths, label_dir, client, model="claude-sonnet-4-20250514",
                      grid_size="2x2", tile_px=512):
    """Verify bounding boxes for a batch of frames.

    Args:
        frame_paths: list of Path to raw frame images
        label_dir: directory containing corresponding .txt label files
        client: anthropic.Anthropic() client instance
        model: API model to use
        grid_size: grid dimensions string
        tile_px: tile size

    Returns:
        list of result dicts
    """
    rows, cols = parse_grid_size(grid_size)
    max_cells = rows * cols

    # Load frames and draw existing boxes on them
    images = []
    paths = []
    for fp in frame_paths[:max_cells]:
        fp = Path(fp)
        img = cv2.imread(str(fp))
        if img is None:
            continue

        # Draw existing pseudo-labels if available
        label_path = Path(label_dir) / fp.with_suffix(".txt").name
        if label_path.exists():
            h, w = img.shape[:2]
            for line in label_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, xc, yc, bw, bh = int(parts[0]), *[float(x) for x in parts[1:]]
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        images.append(img)
        paths.append(fp)

    if not images:
        return []

    composite, mapping = compose_grid(images, paths, parse_grid_size(grid_size), tile_px)
    img_b64 = _encode_image(composite)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": _build_prompt(len(mapping))},
            ],
        }],
    )

    # Parse response
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        return [{"image": paths[i].name, "bbox_quality": "unknown", "certainty": "low",
                 "missing_objects": [], "false_positives": [], "error": "parse_failed"}
                for i in range(len(paths))]

    # Re-attach to source paths
    output = []
    for item in results:
        idx = item.get("cell_index", -1)
        if idx in mapping:
            item["image"] = mapping[idx].name
        output.append(item)

    return output
```

- [ ] **Step 2: Implement brand detection module**

```python
# src/review/brand.py
"""Module 2: Detect vehicle brand/model using Vision API."""
import json
import base64
from pathlib import Path

import cv2

from src.review.grid import compose_grid, parse_grid_size


def _encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode("utf-8")


def _build_prompt(n_cells):
    return f"""You are identifying vehicle brands and models from cropped photos.

This image is a grid of {n_cells} vehicle crops. Each cell has an index number (0-{n_cells-1}) in the top-left.

For each cell, identify the vehicle and return a JSON array of length {n_cells} in row-major order.

Each element must have:
- "cell_index": integer
- "brand": string (e.g. "Toyota", "Hyundai", "Kia")
- "model": string (e.g. "Corolla", "Accent")
- "year_estimate": string range (e.g. "2015-2020") or "unknown"
- "certainty": "high" | "medium" | "low"

If a cell is black/empty or unrecognizable, return {{"cell_index": N, "brand": "unknown", "model": "unknown", "year_estimate": "unknown", "certainty": "low"}}

Return ONLY the JSON array, no other text."""


def detect_brand_batch(crop_paths, client, model="claude-sonnet-4-20250514",
                       grid_size="3x3", tile_px=512):
    """Identify brands for a batch of vehicle crops.

    Returns list of result dicts.
    """
    rows, cols = parse_grid_size(grid_size)
    max_cells = rows * cols

    images = []
    paths = []
    for cp in crop_paths[:max_cells]:
        cp = Path(cp)
        img = cv2.imread(str(cp))
        if img is not None:
            images.append(img)
            paths.append(cp)

    if not images:
        return []

    composite, mapping = compose_grid(images, paths, parse_grid_size(grid_size), tile_px)
    img_b64 = _encode_image(composite)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": _build_prompt(len(mapping))},
            ],
        }],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        return [{"image": paths[i].name, "brand": "unknown", "model": "unknown",
                 "year_estimate": "unknown", "certainty": "low", "error": "parse_failed"}
                for i in range(len(paths))]

    output = []
    for item in results:
        idx = item.get("cell_index", -1)
        if idx in mapping:
            item["image"] = mapping[idx].name
        output.append(item)

    return output
```

- [ ] **Step 3: Implement plate OCR module with hallucination guardrail**

```python
# src/review/plate.py
"""Module 3: OCR plate crops for Arabic + Latin text using Vision API."""
import json
import re
import base64
from pathlib import Path

import cv2

from src.review.grid import compose_grid, parse_grid_size

# Syrian plate format regexes
_PLATE_PATTERNS = [
    re.compile(r'^[\u0600-\u06FF]{2,8}\s?\d{3,7}$'),          # Arabic gov + digits
    re.compile(r'^[A-Z]{2,8}\s?\d{3,7}$'),                     # Latin gov + digits
    re.compile(r'^[\d\u0660-\u0669]{4,8}$'),                    # Arabic-Indic digits only
    re.compile(r'^\d{4,8}$'),                                    # Western digits only
    re.compile(r'^[\u0600-\u06FF]+$'),                           # Arabic text only
]


def _validate_plate_format(text):
    """Check if text matches a known Syrian plate format.
    Returns (is_valid, flag) tuple.
    """
    if not text or not text.strip():
        return False, "unreadable"

    cleaned = text.strip()
    cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', cleaned)  # strip zero-width chars
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    # Try each pattern
    for pattern in _PLATE_PATTERNS:
        if pattern.match(cleaned):
            return True, None

    # For two-line plates, try each line independently
    lines = cleaned.split('\n')
    if len(lines) >= 2:
        has_digit_line = any(re.match(r'^[\d\u0660-\u0669]{3,}$', l.strip()) for l in lines)
        if has_digit_line:
            return True, None

    return False, "hallucination_suspected"


def _encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode("utf-8")


def _build_prompt(n_cells):
    return f"""You are reading Syrian license plates from cropped photos.

This image is a grid of {n_cells} plate crops. Each cell has an index number (0-{n_cells-1}) in the top-left.

Syrian plates typically have Arabic governorate text (e.g. دمشق, حلب) and digits. Some have Latin transliterations.

For each cell, read the plate and return a JSON array of length {n_cells} in row-major order.

Each element must have:
- "cell_index": integer
- "plate_text_original": string (Arabic text as-is)
- "plate_text_latin": string (Latin transliteration/translation)
- "plate_layout_type": "one_line" | "two_line"
- "line_count": integer
- "governorate_text_visible": boolean
- "arabic_visible": boolean
- "latin_visible": boolean
- "plate_color_style": string (e.g. "white_blue", "white_red", "yellow")
- "certainty": "high" | "medium" | "low"

If a cell is black/empty or illegible, return with certainty "low" and plate_text_original "".

Return ONLY the JSON array, no other text."""


def read_plate_batch(crop_paths, client, model="claude-sonnet-4-20250514",
                     grid_size="3x3", tile_px=512):
    """OCR a batch of plate crops.

    Returns list of result dicts with format_valid and format_flag fields added.
    """
    rows, cols = parse_grid_size(grid_size)
    max_cells = rows * cols

    images = []
    paths = []
    for cp in crop_paths[:max_cells]:
        cp = Path(cp)
        img = cv2.imread(str(cp))
        if img is not None:
            images.append(img)
            paths.append(cp)

    if not images:
        return []

    composite, mapping = compose_grid(images, paths, parse_grid_size(grid_size), tile_px)
    img_b64 = _encode_image(composite)

    response = client.messages.create(
        model=model,
        max_tokens=3000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": _build_prompt(len(mapping))},
            ],
        }],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        return [{"image": paths[i].name, "plate_text_original": "", "plate_text_latin": "",
                 "certainty": "low", "format_valid": False, "format_flag": "parse_failed",
                 "error": "parse_failed"}
                for i in range(len(paths))]

    # Post-process: validate formats, re-attach paths
    output = []
    for item in results:
        idx = item.get("cell_index", -1)
        if idx in mapping:
            item["image"] = mapping[idx].name

        # Hallucination guardrail
        is_valid, flag = _validate_plate_format(item.get("plate_text_original", ""))
        item["format_valid"] = is_valid
        item["format_flag"] = flag

        if not is_valid:
            item["certainty"] = "low"  # force low certainty for invalid formats

        output.append(item)

    return output
```

- [ ] **Step 4: Commit**

```bash
git add src/review/bbox.py src/review/brand.py src/review/plate.py
git commit -m "feat: AI review modules for bbox, brand, and plate OCR with guardrails"
```

---

### Task 14: Plate Format Validation Tests

**Files:**
- Create: `tests/test_plate_format.py`

- [ ] **Step 1: Write hallucination guardrail tests**

```python
# tests/test_plate_format.py
from src.review.plate import _validate_plate_format


def test_arabic_gov_digits_valid():
    valid, flag = _validate_plate_format("دمشق 345678")
    assert valid is True
    assert flag is None


def test_arabic_gov_no_space_valid():
    valid, flag = _validate_plate_format("حلب123456")
    assert valid is True


def test_latin_gov_digits_valid():
    valid, flag = _validate_plate_format("DAMASCUS 345678")
    assert valid is True


def test_digits_only_valid():
    valid, flag = _validate_plate_format("345678")
    assert valid is True


def test_arabic_indic_digits_valid():
    valid, flag = _validate_plate_format("٣٤٥٦٧٨")
    assert valid is True


def test_empty_string_invalid():
    valid, flag = _validate_plate_format("")
    assert valid is False
    assert flag == "unreadable"


def test_random_text_invalid():
    valid, flag = _validate_plate_format("Hello World 123 ABC")
    assert valid is False
    assert flag == "hallucination_suspected"


def test_mixed_garbage_invalid():
    valid, flag = _validate_plate_format("XY-123-AB-456")
    assert valid is False
    assert flag == "hallucination_suspected"
```

- [ ] **Step 2: Run tests to verify they pass**

Run: `python -m pytest tests/test_plate_format.py -v`
Expected: 8 passed

- [ ] **Step 3: Commit**

```bash
git add tests/test_plate_format.py
git commit -m "test: plate format validation for hallucination guardrail"
```

---

### Task 15: Review CLI

**Files:**
- Create: `review.py`

- [ ] **Step 1: Implement review.py CLI**

```python
# review.py
"""AI Review CLI — run bbox, brand, or plate review on collected data."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2


def _collect_images(image_arg, dir_arg, extensions=("*.jpg", "*.png")):
    """Collect image paths from --image or --dir arguments."""
    paths = []
    if image_arg:
        from glob import glob
        paths = [Path(p) for p in glob(image_arg)]
    elif dir_arg:
        d = Path(dir_arg)
        for ext in extensions:
            paths.extend(sorted(d.rglob(ext)))
    return paths


def _get_client():
    """Get Anthropic client."""
    import anthropic
    return anthropic.Anthropic()


def _append_results(results, output_file):
    """Append results to JSONL file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _batch(items, size):
    """Yield batches of given size."""
    for i in range(0, len(items), size):
        yield items[i:i + size]


def cmd_bbox(args):
    from src.review.bbox import verify_bbox_batch
    from src.review.grid import parse_grid_size

    paths = _collect_images(args.image, args.dir, ("*.jpg",))
    if not paths:
        print("[review] No images found.")
        return

    client = _get_client()
    rows, cols = parse_grid_size(args.bbox_grid_size)
    batch_size = rows * cols
    output_file = Path(args.output_dir) / "review" / "ai" / "bbox_results.jsonl"

    print(f"[review:bbox] Processing {len(paths)} frames in batches of {batch_size}")
    for i, batch_paths in enumerate(_batch(paths, batch_size)):
        print(f"  Batch {i + 1}...")
        label_dir = batch_paths[0].parent.parent.parent / "labels" / batch_paths[0].parent.name
        results = verify_bbox_batch(batch_paths, label_dir, client,
                                    grid_size=args.bbox_grid_size, tile_px=args.tile_px)
        _append_results(results, output_file)
        time.sleep(1)  # rate limit courtesy

    print(f"[review:bbox] Results saved to {output_file}")


def cmd_brand(args):
    from src.review.brand import detect_brand_batch
    from src.review.grid import parse_grid_size

    paths = _collect_images(args.image, args.dir, ("*.png",))
    if not paths:
        print("[review] No images found.")
        return

    client = _get_client()
    rows, cols = parse_grid_size(args.grid_size)
    batch_size = rows * cols
    output_file = Path(args.output_dir) / "review" / "ai" / "brand_results.jsonl"

    print(f"[review:brand] Processing {len(paths)} crops in batches of {batch_size}")
    for i, batch_paths in enumerate(_batch(paths, batch_size)):
        print(f"  Batch {i + 1}...")
        results = detect_brand_batch(batch_paths, client,
                                     grid_size=args.grid_size, tile_px=args.tile_px)
        _append_results(results, output_file)
        time.sleep(1)

    print(f"[review:brand] Results saved to {output_file}")


def cmd_plate(args):
    from src.review.plate import read_plate_batch
    from src.review.grid import parse_grid_size

    paths = _collect_images(args.image, args.dir, ("*.png",))
    if not paths:
        print("[review] No images found.")
        return

    client = _get_client()
    rows, cols = parse_grid_size(args.grid_size)
    batch_size = rows * cols
    output_file = Path(args.output_dir) / "review" / "ai" / "plate_results.jsonl"
    suspect_file = Path(args.output_dir) / "review" / "queues" / "plate_format_suspect.jsonl"

    print(f"[review:plate] Processing {len(paths)} crops in batches of {batch_size}")
    for i, batch_paths in enumerate(_batch(paths, batch_size)):
        print(f"  Batch {i + 1}...")
        results = read_plate_batch(batch_paths, client,
                                   grid_size=args.grid_size, tile_px=args.tile_px)

        # Split: valid results go to main file, format-suspect go to suspect queue
        valid = [r for r in results if r.get("format_valid", True)]
        suspect = [r for r in results if not r.get("format_valid", True)]

        _append_results(valid, output_file)
        if suspect:
            _append_results(suspect, suspect_file)
            print(f"    {len(suspect)} format-suspect items queued for review")

        time.sleep(1)

    print(f"[review:plate] Results saved to {output_file}")


def cmd_all(args):
    print("[review] Running all three modules...")
    base = Path(args.output_dir)

    args.dir = str(base / "raw" / "frames")
    args.image = None
    cmd_bbox(args)

    args.dir = str(base / "raw" / "crops" / "vehicles")
    cmd_brand(args)

    args.dir = str(base / "raw" / "crops" / "plates")
    cmd_plate(args)


def main():
    p = argparse.ArgumentParser(description="AI Review Pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    # Common args
    for name, func in [("bbox", cmd_bbox), ("brand", cmd_brand),
                        ("plate", cmd_plate), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.add_argument("--image", default=None, help="Glob pattern for images")
        sp.add_argument("--dir", default=None, help="Directory of images")
        sp.add_argument("--output-dir", default="./output", help="Output directory")
        sp.add_argument("--grid-size", default="3x3", help="Crops per composite")
        sp.add_argument("--bbox-grid-size", default="2x2", help="Frames per composite (bbox only)")
        sp.add_argument("--tile-px", type=int, default=512, help="Per-cell tile size")
        sp.set_defaults(func=func)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add review.py
git commit -m "feat: AI review CLI with bbox, brand, plate commands and grid batching"
```

---

## Phase 1.5b — Human Review Tool

### Task 16: Human Review CLI

**Files:**
- Create: `human_review.py`

- [ ] **Step 1: Implement human_review.py**

```python
# human_review.py
"""Human review tool for AI-flagged items."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2


def _load_review_queue(review_dir, module):
    """Load items needing human review (certainty = medium or low)."""
    results_file = Path(review_dir) / "ai" / f"{module}_results.jsonl"
    items = []
    if results_file.exists():
        for line in results_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("certainty") in ("medium", "low"):
                items.append((module, item))

    # Also load format suspects for plate module
    if module == "plate":
        suspect_file = Path(review_dir) / "queues" / "plate_format_suspect.jsonl"
        if suspect_file.exists():
            for line in suspect_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                items.append(("plate_suspect", json.loads(line)))

    return items


def _find_image(image_name, raw_dir):
    """Find the full path to an image by name in raw directory."""
    for p in Path(raw_dir).rglob(image_name):
        return p
    return None


def _display_info(module, item):
    """Print AI label info for the reviewer."""
    print("\n" + "=" * 60)
    print(f"Module: {module} | Image: {item.get('image', '?')}")
    print(f"Certainty: {item.get('certainty', '?')}")

    if module in ("brand",):
        print(f"Brand: {item.get('brand', '?')} | Model: {item.get('model', '?')}")
        print(f"Year: {item.get('year_estimate', '?')}")
    elif module in ("plate", "plate_suspect"):
        print(f"Text (original): {item.get('plate_text_original', '?')}")
        print(f"Text (latin): {item.get('plate_text_latin', '?')}")
        print(f"Layout: {item.get('plate_layout_type', '?')}")
        if item.get("format_flag"):
            print(f"FORMAT FLAG: {item['format_flag']}")
    elif module == "bbox":
        print(f"BBox quality: {item.get('bbox_quality', '?')}")
        if item.get("missing_objects"):
            print(f"Missing: {item['missing_objects']}")
        if item.get("false_positives"):
            print(f"False positives: {item['false_positives']}")

    print("-" * 60)
    print("[y] approve | [n] reject | [e] edit | [s] skip")


def _promote(module, item, raw_dir, output_dir, correction=None):
    """Copy approved item to approved/ directory."""
    image_name = item.get("image", "")
    src = _find_image(image_name, raw_dir)
    if not src:
        print(f"  WARNING: source image not found: {image_name}")
        return

    if module == "bbox":
        dest_dir = Path(output_dir) / "approved" / "detector" / "images"
    elif module in ("plate", "plate_suspect"):
        dest_dir = Path(output_dir) / "approved" / "ocr" / "plate_crops"
    elif module == "brand":
        dest_dir = Path(output_dir) / "approved" / "vehicle_classifier" / "vehicle_crops"
    else:
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)

    # Write label if applicable
    if module in ("plate", "plate_suspect"):
        label_file = Path(output_dir) / "approved" / "ocr" / "plate_labels.jsonl"
        label_data = {k: v for k, v in item.items()}
        if correction:
            label_data["plate_text_original"] = correction
            label_data["human_corrected"] = True
        label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(label_file, "a") as f:
            f.write(json.dumps(label_data, ensure_ascii=False) + "\n")
    elif module == "brand":
        label_file = Path(output_dir) / "approved" / "vehicle_classifier" / "brand_labels.jsonl"
        label_data = {k: v for k, v in item.items()}
        if correction:
            label_data["brand"] = correction
            label_data["human_corrected"] = True
        label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(label_file, "a") as f:
            f.write(json.dumps(label_data, ensure_ascii=False) + "\n")


def _reject(module, item, raw_dir, output_dir):
    """Copy rejected item to rejected/ directory."""
    image_name = item.get("image", "")
    src = _find_image(image_name, raw_dir)
    if not src:
        return

    if module == "bbox":
        dest_dir = Path(output_dir) / "rejected" / "detector"
    elif module in ("plate", "plate_suspect"):
        dest_dir = Path(output_dir) / "rejected" / "ocr"
    elif module == "brand":
        dest_dir = Path(output_dir) / "rejected" / "vehicle_classifier"
    else:
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)


def run_review(review_dir, raw_dir, output_dir):
    """Main review loop."""
    all_items = []
    for module in ("bbox", "brand", "plate"):
        all_items.extend(_load_review_queue(review_dir, module))

    if not all_items:
        print("[review] No items to review.")
        return

    print(f"[review] {len(all_items)} items to review")

    review_log_path = Path(review_dir) / "human" / "review_log.jsonl"
    review_log_path.parent.mkdir(parents=True, exist_ok=True)

    reviewed = 0
    for module, item in all_items:
        image_name = item.get("image", "")
        img_path = _find_image(image_name, raw_dir)

        if img_path:
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imshow("Review", img)
                cv2.waitKey(100)

        _display_info(module, item)

        while True:
            action = input("Action: ").strip().lower()
            if action in ("y", "n", "e", "s"):
                break
            print("Invalid. Use y/n/e/s.")

        correction = None
        if action == "e":
            correction = input("Correction: ").strip()
            action = "y"  # edit = approve with correction

        # Log decision
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "image": image_name,
            "action": {"y": "approved", "n": "rejected", "s": "skipped"}[action],
            "correction": correction,
            "original_certainty": item.get("certainty"),
        }
        with open(review_log_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if action == "y":
            _promote(module, item, raw_dir, output_dir, correction)
            print("  -> APPROVED")
        elif action == "n":
            _reject(module, item, raw_dir, output_dir)
            print("  -> REJECTED")
        else:
            print("  -> SKIPPED")

        reviewed += 1

    cv2.destroyAllWindows()
    print(f"[review] Done. Reviewed {reviewed} items.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Human Review Tool")
    p.add_argument("--review-dir", default="./output/review", help="Review directory")
    p.add_argument("--raw-dir", default="./output/raw", help="Raw data directory")
    p.add_argument("--output-dir", default="./output", help="Output directory")
    args = p.parse_args()
    run_review(args.review_dir, args.raw_dir, args.output_dir)
```

- [ ] **Step 2: Commit**

```bash
git add human_review.py
git commit -m "feat: human review CLI with approve/reject/edit/skip actions"
```

---

## Task Summary

| # | Task | Phase | Files | Tests |
|---|------|-------|-------|-------|
| 1 | Project setup + config | 1 | `requirements.txt`, `src/config.py`, `tests/conftest.py` | `test_config.py` (2) |
| 2 | Camera capture + frame skip | 1 | `src/capture.py` | `test_capture.py` (2) |
| 3 | Vehicle + plate detector | 1 | `src/detector.py` | integration only |
| 4 | Multi-object tracker | 1 | `src/tracker.py` | `test_tracker.py` (5) |
| 5 | Vehicle-plate association | 1 | `src/association.py` | `test_association.py` (5) |
| 6 | Quality gate | 1 | `src/quality.py` | `test_quality.py` (5) |
| 7 | Save decision + file I/O | 1 | `src/saver.py` | `test_saver.py` (6) |
| 8 | Metadata logger | 1 | `src/metadata.py` | `test_metadata.py` (5) |
| 9 | Live display | 1 | `src/display.py` | visual only |
| 10 | Main pipeline orchestrator | 1 | `src/collector.py` | integration only |
| 11 | Pre-flight validation | 1 | `preflight.py` | integration only |
| 12 | Grid compositor | 1.5 | `src/review/grid.py` | `test_review.py` (3) |
| 13 | AI review modules | 1.5 | `src/review/{bbox,brand,plate}.py` | — |
| 14 | Plate format tests | 1.5 | — | `test_plate_format.py` (8) |
| 15 | Review CLI | 1.5 | `review.py` | — |
| 16 | Human review CLI | 1.5b | `human_review.py` | — |

**Total: 16 tasks, ~41 tests**
