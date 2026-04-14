# Syrian License Plate Data Collector — Design Spec v2

## Purpose

A Python-based **data collection pipeline** that uses a live camera to detect vehicles and license plates in real time, saving high-value training data for:

1. Fine-tuning a detector for `vehicle` and `plate`
2. Collecting cropped plate images for future Arabic/Latin OCR training
3. Collecting cropped vehicle images for future brand/model classifier training
4. Preserving metadata needed for later Jetson deployment, review, and dataset cleanup

Target deployment remains **Jetson Orin Nano (8GB)** in police cars, but **Phase 1 runs on Mac** as a dedicated data collection tool.

---

## Scope

- Syrian license plates (Arabic + Latin characters)
- Live USB webcam or future CSI camera feed
- Real-time detection with live display
- Smarter deduplication than simple frame saving
- Training-quality crops and high-quality saved frames
- Vehicle–plate association with confidence scoring
- Raw/approved dataset separation from the beginning

---

## Key Design Principles

1. **Phase 1 is a collector, not the final production stack**
   - Use two detectors now if that improves collection quality.
   - Do not prematurely force the final single-model production pipeline into the collection stage.

2. **Save metadata-rich raw data first**
   - Pseudo-labels are useful, but they are not ground truth.
   - Every saved object must carry enough metadata to support review, filtering, and retraining.

3. **Optimize for future edge deployment**
   - Structure outputs so they can later feed a TensorRT / DeepStream-style Jetson deployment.

4. **Prefer quality over quantity**
   - Do not save every detected frame.
   - Save when novelty, quality, and usefulness justify it.

---

## Recommended High-Level Architecture

```text
Camera Input (USB / CSI)
    │
    ▼
Full-Resolution Capture
    │
    ├──▶ Resize Copy for Inference
    │        ├──▶ Vehicle Detector (YOLO26)
    │        └──▶ Plate Detector (YOLO-format model)
    │
    ▼
Tracking Layer
    │   ├── Vehicle tracks
    │   └── Plate tracks
    │
    ▼
Association Engine
    │   └── Scored vehicle ↔ plate matching
    │
    ▼
Quality Gate
    │   ├── Blur / sharpness check
    │   ├── Size thresholds
    │   ├── Truncation check
    │   ├── Exposure sanity check
    │   └── Save-worthiness score
    │
    ▼
Save Decision Engine
    │   ├── New object
    │   ├── Cooldown expired
    │   ├── Better-quality replacement save
    │   └── Hard-negative sampling
    │
    ├──▶ Raw Frame Save
    ├──▶ Raw Vehicle Crop Save
    ├──▶ Raw Plate Crop Save
    ├──▶ Pseudo-label Save
    └──▶ Metadata JSONL Save
```

---

## Detection Models

### 1. Vehicle Detection

- Model family: **Ultralytics YOLO26 detect**
- Initial training/inference target classes during collection metadata:
  - `car`
  - `truck`
  - `bus`
  - optional `unknown_vehicle`
- Recommended detector training class collapse for first custom model:
  - `vehicle`
  - `plate`

### 2. Plate Detection

- Model family: separate YOLO-format plate detector
- Runs on the same frame in parallel with vehicle detection
- Use full-frame inference in Phase 1
- Specific plate model can be selected during implementation after testing on Syrian data

### 3. Confidence Defaults

- `vehicle_conf = 0.30`
- `plate_conf = 0.30`

Low thresholds are acceptable for collection, because review/filtering comes later.

---

## Tracking Policy

The original centroid-only concept is replaced in v2 with a stronger but still lightweight design.

### Preferred approach

Use tracker-backed object IDs from the inference pipeline.

### Acceptable fallback

If external tracking is not used, matching between frames should combine:
- centroid distance
- IoU overlap
- box size similarity
- class consistency

### Track record fields

Each active track should store:
- `track_id`
- `object_type`
- `first_seen_ts`
- `last_seen_ts`
- `last_saved_ts`
- `save_count`
- `best_quality_score`
- `best_crop_path`

### Stale track removal

- Default timeout: `4.0 seconds`
- Remove tracks that have not matched within timeout

---

## Vehicle–Plate Association Policy

The v1 rule “plate inside car bbox” is replaced with **scored association**.

### Candidate generation

A plate may be considered a candidate child of a vehicle if any of these are true:
- plate center lies inside vehicle box
- plate box overlaps vehicle box above a minimum threshold
- plate lies near the lower region of a plausible vehicle box

### Association score components

Compute a score using:
- containment score
- overlap score
- distance from plate center to vehicle bottom-center
- preference for plate center being in lower half of the vehicle box
- relative size plausibility

### Output states

Each plate must end as one of:
- `matched`
- `ambiguous`
- `unmatched`

Do **not** force ambiguous matches.

---

## Quality Gate

Only save when detections are useful for training.

### Per-crop or per-object checks

Calculate:
- blur / sharpness score
- crop width and height in pixels
- crop area in pixels
- truncation at image boundaries
- basic exposure sanity
- optional occlusion heuristic

### Save-worthiness rules

A crop is eligible only if it passes minimum thresholds such as:
- minimum plate width / height
- minimum vehicle area
- blur score above threshold
- crop not excessively truncated

### Better-save replacement rule

Even if the same track was already saved, save again when the new observation is materially better, for example:
- sharper plate crop
- larger visible vehicle crop
- less truncation
- clearer rear/front view

---

## Save Decision Logic

Save when **any** of the following is true:

1. A new track appears
2. Cooldown has expired and quality is acceptable
3. The current observation is significantly better than the last saved version of the same track
4. The observation qualifies as a useful hard negative

### Default cooldowns

- `vehicle_cooldown = 3.0 sec`
- `plate_cooldown = 2.0 sec`

---

## Hard-Negative Sampling

Phase 1 should intentionally preserve some negative or difficult examples.

Examples:
- vehicle with no visible plate
- partial plate
- blurred plate
- sign or sticker mistaken for plate
- crowded overlapping vehicles
- far-distance vehicles
- low-light difficult cases

These should be marked explicitly in metadata so they can be used later for detector robustness.

---

## Image Saving Policy

### 1. Raw Full Frames

Save only when at least one crop-worthy or hard-negative event is triggered.

- Format: **JPEG**, quality 95
- Path:
  `raw/frames/YYYY-MM-DD/frm_{session_id}_{timestamp_ms}_{frame_idx}.jpg`

### 2. Raw Vehicle Crops

- Source: full-resolution original frame
- Padding: default 10%
- Format: **PNG**
- Path:
  `raw/crops/vehicles/YYYY-MM-DD/veh_{session_id}_{track_id}_{timestamp_ms}.png`

### 3. Raw Plate Crops

- Source: full-resolution original frame
- Padding: default 5%
- Format: **PNG**
- Path:
  `raw/crops/plates/YYYY-MM-DD/plt_{session_id}_{track_id}_{timestamp_ms}.png`

### 4. Pseudo-Labels

- Save YOLO-format pseudo-label file for each saved raw frame
- Format:
  `class_id x_center y_center width height`
- Detector class mapping for saved labels:
  - `0 = vehicle`
  - `1 = plate`

### 5. Important Rule

Pseudo-labels must always be treated as **unverified** until approved through review.

---

## Metadata Logging

Use JSONL as the canonical event log.

### Primary metadata file

`raw/metadata/save_events.jsonl`

### Recommended additional metadata files

- `raw/metadata/detections.jsonl`
- `raw/metadata/tracks.jsonl`
- `raw/metadata/sessions.jsonl`

### Example save event schema

```json
{
  "event_id": "evt_000001",
  "session_id": "sess_20260414_cam0_a1b2",
  "timestamp": "2026-04-14T14:30:22.481Z",
  "camera_id": "cam0",
  "frame_path": "raw/frames/2026-04-14/frm_sess_20260414_cam0_a1b2_1713105022481_000123.jpg",
  "label_path": "raw/labels/2026-04-14/frm_sess_20260414_cam0_a1b2_1713105022481_000123.txt",
  "image_width": 1920,
  "image_height": 1080,
  "detections": [
    {
      "object_type": "vehicle",
      "track_id": "veh_17",
      "vehicle_type": "car",
      "bbox_xyxy": [120, 80, 450, 320],
      "detector_conf": 0.81,
      "crop_path": "raw/crops/vehicles/2026-04-14/veh_sess_20260414_cam0_a1b2_veh_17_1713105022481.png",
      "quality_score": 0.86,
      "blur_score": 142.2,
      "truncated": false,
      "occluded": false,
      "associated_plate_track_id": "plt_44"
    },
    {
      "object_type": "plate",
      "track_id": "plt_44",
      "bbox_xyxy": [200, 280, 340, 310],
      "detector_conf": 0.77,
      "crop_path": "raw/crops/plates/2026-04-14/plt_sess_20260414_cam0_a1b2_plt_44_1713105022481.png",
      "quality_score": 0.74,
      "blur_score": 119.5,
      "truncated": false,
      "occluded": false,
      "association_status": "matched",
      "associated_vehicle_track_id": "veh_17",
      "association_score": 0.92
    }
  ]
}
```

### Recommended metadata fields per detection

- `track_id`
- `object_type`
- `bbox_xyxy`
- `detector_conf`
- `quality_score`
- `blur_score`
- `truncated`
- `occluded`
- `crop_path`
- `association_status`
- `association_score`
- `associated_vehicle_track_id`
- `associated_plate_track_id`
- `negative_type` (if applicable)
- `review_status`

---

## Output Directory Structure v2

```text
output/
├── raw/
│   ├── frames/
│   │   └── YYYY-MM-DD/
│   ├── labels/
│   │   └── YYYY-MM-DD/
│   ├── crops/
│   │   ├── vehicles/
│   │   │   └── YYYY-MM-DD/
│   │   └── plates/
│   │       └── YYYY-MM-DD/
│   ├── metadata/
│   │   ├── save_events.jsonl
│   │   ├── detections.jsonl
│   │   ├── tracks.jsonl
│   │   └── sessions.jsonl
│   └── classes.txt
├── review/
│   ├── ai/
│   ├── human/
│   └── queues/
├── approved/
│   ├── detector/
│   ├── ocr/
│   └── vehicle_classifier/
└── rejected/
    ├── detector/
    ├── ocr/
    └── vehicle_classifier/
```

---

## Live Display

- OpenCV preview window during Phase 1
- Green boxes for vehicles
- Red boxes for plates
- Optional lines between matched plate and parent vehicle
- Overlay suggested counters:
  - vehicles detected
  - plates detected
  - active tracks
  - saved events
  - rejected-by-quality count
- Press `q` to quit cleanly

---

## Session Statistics

Maintain session-level stats file or session JSONL records including:
- session start time
- session end time
- duration
- total frames processed
- total detections
- total saved events
- saves by object type
- rejects by blur
- rejects by size
- rejects by cooldown
- matched plates
- unmatched plates
- ambiguous matches

---

## Review Pipeline v2

### Phase 1.5 — AI Review

Keep the three review modules, but define them as **triage tools**, not final truth.

#### Module 1: bbox review
Input:
- raw full frame
- pseudo-label file

Output:
- bbox quality assessment
- missing/false-positive notes
- certainty score

#### Module 2: vehicle label suggestion
Input:
- vehicle crop

Output:
- brand suggestion
- model suggestion
- year range estimate
- certainty score

#### Module 3: plate text suggestion
Input:
- plate crop

Output:
- Arabic text guess
- Latin transliteration guess
- format notes
- certainty score

### Confidence policy

- `high` = low review priority
- `medium` = human review recommended
- `low` = human review required

Do not assume “high” means guaranteed correct.

---

## Human Review Tool v2

A human review CLI should operate on flagged review items and promote data into approved datasets.

### Actions

For each flagged item:
- `y` approve
- `n` reject
- `e` edit
- `s` skip

### Promotion targets

Approved reviewed data should move or copy into:
- `approved/detector/`
- `approved/ocr/`
- `approved/vehicle_classifier/`

Rejected items should move or copy into:
- `rejected/detector/`
- `rejected/ocr/`
- `rejected/vehicle_classifier/`

---

## OCR Strategy v2

### Short term

Use cloud vision + baseline OCR as labeling aids.

### Long term

Treat the production OCR engine as a separate evaluation decision.

The project should preserve enough metadata and reviewed plate crops so the OCR stage can later compare:
- EasyOCR baseline
- custom OCR model
- LPR-style model deployment on Jetson

### OCR metadata to preserve now

- `plate_layout_type`
- `line_count`
- `governorate_text_visible`
- `arabic_visible`
- `latin_visible`
- `plate_color_style`

---

## Configuration

Recommended config options for v2:

| Parameter | Default | Description |
| --- | --- | --- |
| `--camera` | `0` | Camera device index |
| `--vehicle-model` | `yolo26n.pt` | Vehicle detector model |
| `--plate-model` | required | Plate detector model path |
| `--vehicle-conf` | `0.30` | Vehicle detection threshold |
| `--plate-conf` | `0.30` | Plate detection threshold |
| `--imgsz` | `960` | Inference image size |
| `--output` | `./output` | Output directory |
| `--vehicle-cooldown` | `3.0` | Seconds between vehicle saves |
| `--plate-cooldown` | `2.0` | Seconds between plate saves |
| `--track-timeout` | `4.0` | Track stale timeout |
| `--min-plate-w` | `40` | Minimum plate width in pixels |
| `--min-plate-h` | `14` | Minimum plate height in pixels |
| `--min-vehicle-area` | `12000` | Minimum vehicle crop area |
| `--min-blur-score` | configurable | Blur rejection threshold |
| `--save-better-only` | `true` | Save again if new crop is better |
| `--save-hard-negatives` | `true` | Keep useful negatives |
| `--no-display` | `false` | Run headless |
| `--session-id` | auto | Session identifier |

---

## Dependencies

### Phase 1 — Collection

```text
ultralytics
opencv-python
numpy
```

### Phase 1.5 — Review

```text
anthropic or google-generativeai
Pillow
```

### Optional future deployment stack

- TensorRT export path
- DeepStream integration path
- Jetson-specific inference benchmarking

---

## Final Deployment Target

Police-car-mounted Jetson Orin Nano system with:
- one or more cameras depending on validated bandwidth and FPS
- detector for `vehicle` + `plate`
- lightweight vehicle brand/model classifier
- OCR for Arabic + Latin Syrian plate text
- reliable vehicle–plate association
- optimized edge inference

---

## Revised Phased Roadmap

| Phase | Hardware | What | Models / Stack |
| --- | --- | --- | --- |
| **1 - Collect** | Mac | Real-time data collection from single camera | YOLO26 vehicle detector + plate detector |
| **1.5 - AI Review** | Cloud API | Label triage and suggestion | Claude / Gemini Vision + OCR baseline |
| **1.5b - Human Review** | Mac | Approve, reject, or correct labels | Python review CLI |
| **2 - Train** | Mac / Cloud | Train approved detector, OCR, and classifier data | YOLO26 custom detector + vehicle classifier + OCR training |
| **3 - Deploy** | Jetson Orin Nano | Edge inference pipeline in police car | TensorRT / DeepStream-compatible deployment |

---

## What Changed From v1

1. Centroid-only dedup was replaced by stronger tracking logic.
2. Plate-in-car association was replaced by scored matching.
3. Image quality gating was added.
4. Raw and approved data were separated.
5. Metadata was expanded substantially.
6. Hard-negative sampling was added.
7. Detector training labels were simplified to `vehicle` and `plate`.
8. OCR was reframed as a staged decision rather than a fixed final tool.

---

## Out of Scope for Phase 1

- Final OCR deployment
- Final vehicle brand/model classifier training
- Database storage
- Web interface
- Production multi-camera synchronization
- Full DeepStream implementation
- Final model fine-tuning scripts

