# Syrian License Plate Data Collector — Final Design Spec

## Purpose

A Python-based **data collection pipeline** that uses a live USB webcam to detect vehicles and license plates in real time, saving high-value training data for:

1. Fine-tuning a single YOLO26 model to detect both vehicles and plates
2. Collecting cropped plate images for future Arabic/Latin OCR training
3. Collecting cropped vehicle images for car brand/model classifier training
4. Preserving metadata needed for later Jetson deployment, review, and dataset cleanup

Target deployment: **Jetson Orin Nano (8GB)** in police cars — Phase 1 runs on Mac for data collection.

---

## Scope

- Syrian license plates (Arabic + Latin characters)
- Live USB webcam feed with real-time display
- Smarter deduplication with quality-aware save decisions
- Training-quality crops (PNG) and high-quality saved frames (JPEG 95%)
- Vehicle-plate association with scored matching
- Raw/approved dataset separation from the beginning
- Hard-negative sampling for model robustness

---

## Key Design Principles

1. **Phase 1 is a collector, not the final production stack**
   - Use two detectors now (YOLO26x for vehicles + plate model) for best collection quality.
   - Do not force the final single-model pipeline into the collection stage.

2. **Save metadata-rich raw data first**
   - Pseudo-labels are useful but are not ground truth.
   - Every saved object must carry enough metadata for review, filtering, and retraining.

3. **Optimize for future edge deployment**
   - Structure outputs so they can later feed a TensorRT / DeepStream Jetson deployment.

4. **Prefer quality over quantity**
   - Do not save every detected frame.
   - Save when novelty, quality, and usefulness justify it.

---

## Architecture

```text
USB Webcam
    │
    ▼
Full-Resolution Capture
    │
    ├──▶ Resize Copy for Inference
    │        ├──▶ Vehicle Detector (YOLO26x, COCO pre-trained, conf 0.3)
    │        └──▶ Plate Detector (YOLO-format model, conf 0.3)
    │               (both run in parallel on same frame)
    │
    ▼
Tracking Layer
    │   ├── Vehicle tracks (centroid + IoU + size + class)
    │   └── Plate tracks
    │
    ▼
Association Engine
    │   └── Scored vehicle ↔ plate matching
    │         (matched / ambiguous / unmatched)
    │
    ▼
Quality Gate
    │   ├── Blur / sharpness check
    │   ├── Size thresholds (min plate w/h, min vehicle area)
    │   ├── Truncation check (at image boundaries)
    │   ├── Exposure sanity check
    │   └── Save-worthiness score
    │
    ▼
Save Decision Engine
    │   ├── New object → save
    │   ├── Cooldown expired + quality OK → save
    │   ├── Better-quality replacement → resave
    │   └── Hard-negative → save with negative flag
    │
    ├──▶ Raw Frame Save (JPEG 95%)
    ├──▶ Raw Vehicle Crop Save (PNG)
    ├──▶ Raw Plate Crop Save (PNG)
    ├──▶ Pseudo-label Save (YOLO format)
    ├──▶ Metadata JSONL Save
    │
    └──▶ Live Display (OpenCV window with boxes + counters)
```

---

## Detection Models

### 1. Vehicle Detection

- Model: **YOLO26x** pre-trained on COCO (`yolo26x.pt`)
- COCO classes used: `car`, `truck`, `bus`
- Collapsed to single class `vehicle` in saved labels
- Confidence threshold: 0.3

### 2. Plate Detection

- Model: separate YOLO-format plate detector (specific model selected during implementation after testing on Syrian data)
- Runs on **full frame** in parallel with vehicle detection
- Confidence threshold: 0.3

Low thresholds are acceptable for collection — review/filtering comes later.

---

## Tracking Policy

### Matching between frames

Combine multiple signals for robust tracking:

- Centroid distance
- IoU overlap
- Box size similarity
- Class consistency

### Track record fields

Each active track stores:

- `track_id`
- `object_type` (vehicle / plate)
- `first_seen_ts`
- `last_seen_ts`
- `last_saved_ts`
- `save_count`
- `best_quality_score`
- `best_crop_path`

### Stale track removal

- Default timeout: 4.0 seconds
- Remove tracks that have not matched within timeout

---

## Vehicle-Plate Association

### Scored matching (not simple containment)

A plate is a candidate child of a vehicle if any of these are true:

- Plate center lies inside vehicle box
- Plate box overlaps vehicle box above minimum threshold
- Plate lies near the lower region of a plausible vehicle box

### Score components

- Containment score
- Overlap score
- Distance from plate center to vehicle bottom-center
- Preference for plate center in lower half of vehicle box
- Relative size plausibility

### Output states

Each plate ends as one of:

- `matched` — clearly belongs to one vehicle
- `ambiguous` — could belong to multiple vehicles (do not force)
- `unmatched` — no parent vehicle found (still saved)

---

## Quality Gate

Only save when detections are useful for training.

### Per-crop checks

- Blur / sharpness score (Laplacian variance)
- Crop width and height in pixels
- Crop area in pixels
- Truncation at image boundaries
- Basic exposure sanity

### Save-worthiness thresholds

- Minimum plate width: 40px
- Minimum plate height: 14px
- Minimum vehicle area: 12,000px
- Blur score above configurable threshold
- Crop not excessively truncated (>30% outside frame = reject)

### Better-save replacement

Even if the same track was already saved, save again when the new observation is materially better:

- Sharper plate crop
- Larger visible vehicle crop
- Less truncation
- Clearer view

---

## Save Decision Logic

Save when **any** of the following is true:

1. A new track appears and passes quality gate
2. Cooldown has expired and quality is acceptable
3. Current observation is significantly better than last saved version
4. Observation qualifies as a useful hard negative

### Default cooldowns

- Vehicle cooldown: 3.0 seconds
- Plate cooldown: 2.0 seconds

---

## Hard-Negative Sampling

Phase 1 intentionally preserves difficult examples for training robustness:

- Vehicle with no visible plate
- Partial plate
- Blurred plate
- Sign or sticker mistaken for plate
- Crowded overlapping vehicles
- Far-distance vehicles
- Low-light difficult cases

Marked explicitly in metadata with `negative_type` field.

---

## Image Saving Policy

### 1. Raw Full Frames

Save only when at least one crop-worthy or hard-negative event triggers.

- Format: **JPEG 95%**
- Path: `raw/frames/YYYY-MM-DD/frm_{session_id}_{timestamp_ms}_{frame_idx}.jpg`

### 2. Raw Vehicle Crops

- Source: full-resolution original frame
- Padding: 10%
- Format: **PNG**
- Path: `raw/crops/vehicles/YYYY-MM-DD/veh_{session_id}_{track_id}_{timestamp_ms}.png`

### 3. Raw Plate Crops

- Source: full-resolution original frame
- Padding: 5%
- Format: **PNG**
- Path: `raw/crops/plates/YYYY-MM-DD/plt_{session_id}_{track_id}_{timestamp_ms}.png`

### 4. Pseudo-Labels

- YOLO-format label file per saved raw frame
- Format: `class_id x_center y_center width height` (normalized 0-1)
- Class mapping: `0 = vehicle`, `1 = plate`
- Path: `raw/labels/YYYY-MM-DD/frm_{session_id}_{timestamp_ms}_{frame_idx}.txt`

**Important:** Pseudo-labels are unverified until approved through review pipeline.

---

## Metadata Logging

JSONL as canonical event log format.

### Metadata files

- `raw/metadata/save_events.jsonl` — primary event log
- `raw/metadata/tracks.jsonl` — track lifecycle records
- `raw/metadata/sessions.jsonl` — session start/end stats

### Save event schema

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
      "crop_path": "raw/crops/vehicles/2026-04-14/veh_sess_a1b2_veh17_1713105022481.png",
      "quality_score": 0.86,
      "blur_score": 142.2,
      "truncated": false,
      "occluded": false,
      "associated_plate_track_id": "plt_44",
      "negative_type": null,
      "review_status": "pending"
    },
    {
      "object_type": "plate",
      "track_id": "plt_44",
      "bbox_xyxy": [200, 280, 340, 310],
      "detector_conf": 0.77,
      "crop_path": "raw/crops/plates/2026-04-14/plt_sess_a1b2_plt44_1713105022481.png",
      "quality_score": 0.74,
      "blur_score": 119.5,
      "truncated": false,
      "occluded": false,
      "association_status": "matched",
      "associated_vehicle_track_id": "veh_17",
      "association_score": 0.92,
      "negative_type": null,
      "review_status": "pending"
    }
  ]
}
```

---

## Output Directory Structure

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
│   │   ├── tracks.jsonl
│   │   └── sessions.jsonl
│   └── classes.txt                # "vehicle\nplate"
├── review/
│   ├── ai/
│   │   ├── bbox_results.jsonl
│   │   ├── brand_results.jsonl
│   │   └── plate_results.jsonl
│   ├── human/
│   │   └── review_log.jsonl
│   └── queues/
│       ├── bbox_queue.jsonl
│       ├── brand_queue.jsonl
│       └── plate_queue.jsonl
├── approved/
│   ├── detector/
│   │   ├── images/
│   │   └── labels/
│   ├── ocr/
│   │   ├── plate_crops/
│   │   └── plate_labels.jsonl
│   └── vehicle_classifier/
│       ├── vehicle_crops/
│       └── brand_labels.jsonl
└── rejected/
    ├── detector/
    ├── ocr/
    └── vehicle_classifier/
```

---

## Live Display

- OpenCV preview window
- Green boxes for vehicles with label "VEHICLE"
- Red boxes for plates with label "PLATE"
- Lines connecting matched plate to parent vehicle
- Overlay counters:
  - Vehicles detected / Plates detected
  - Active tracks
  - Saved events
  - Rejected-by-quality count
- Press `q` to quit cleanly

---

## Session Statistics

Session-level stats saved to `raw/metadata/sessions.jsonl`:

- Session start/end time and duration
- Total frames processed
- Total detections (vehicle / plate)
- Total saved events
- Saves by object type
- Rejects by blur / size / cooldown
- Matched / unmatched / ambiguous plates

---

## Phase 1.5 — AI Review Pipeline

Three **separate, independent** Python modules — each callable on demand on any image.

### Module 1: `verify_bbox(image_path)`

Verify bounding box accuracy on a full frame + pseudo-label.

- Input: raw full frame + YOLO label file
- Sends to Claude/Gemini Vision API
- Output: appends to `review/ai/bbox_results.jsonl`

```json
{
  "image": "frm_sess_a1b2_1713105022481_000123.jpg",
  "bbox_quality": "good",
  "missing_objects": [],
  "false_positives": [],
  "certainty": "high"
}
```

### Module 2: `detect_brand(car_crop_path)`

Identify vehicle brand and model from a vehicle crop.

- Input: single vehicle crop image
- Sends to Claude/Gemini Vision API
- Output: appends to `review/ai/brand_results.jsonl`

```json
{
  "image": "veh_sess_a1b2_veh17_1713105022481.png",
  "brand": "Toyota",
  "model": "Corolla",
  "year_estimate": "2015-2020",
  "certainty": "high"
}
```

### Module 3: `read_plate(plate_crop_path)`

OCR a plate crop for Arabic + Latin text.

- Input: single plate crop image
- Sends to Claude/Gemini Vision API
- Output: appends to `review/ai/plate_results.jsonl`

```json
{
  "image": "plt_sess_a1b2_plt44_1713105022481.png",
  "plate_text_original": "دمشق ٣٤٥٦٧٨",
  "plate_text_latin": "Damascus 345678",
  "plate_layout_type": "two_line",
  "line_count": 2,
  "governorate_text_visible": true,
  "arabic_visible": true,
  "latin_visible": true,
  "plate_color_style": "white_blue",
  "certainty": "medium"
}
```

### Certainty levels

- **high** — auto-approved, low review priority
- **medium** — human review recommended
- **low** — human review required

"High" does not mean guaranteed correct — spot-checks are still valuable.

### CLI Usage

```bash
# Run individually on a single image
python review.py bbox --image raw/frames/2026-04-14/frm_*.jpg
python review.py brand --image raw/crops/vehicles/2026-04-14/veh_*.png
python review.py plate --image raw/crops/plates/2026-04-14/plt_*.png

# Run on entire folder
python review.py bbox --dir raw/frames/
python review.py brand --dir raw/crops/vehicles/
python review.py plate --dir raw/crops/plates/

# Run all three on everything
python review.py all --output-dir output/
```

---

## Phase 1.5b — Human Review Tool

Python CLI tool for reviewing AI-flagged items.

### Usage

```bash
python human_review.py --review-dir review/ --raw-dir raw/
```

### Per-item actions

For each flagged item (certainty = medium or low):

1. Opens image in OpenCV window
2. Shows AI label (brand, plate text, or bbox quality)
3. Human presses:
   - `y` — approve as-is
   - `n` — reject
   - `e` — edit (type corrected label)
   - `s` — skip for later

### Promotion targets

- Approved detector data → `approved/detector/`
- Approved OCR data → `approved/ocr/`
- Approved brand data → `approved/vehicle_classifier/`
- Rejected data → `rejected/{detector,ocr,vehicle_classifier}/`

### Review log

All decisions logged to `review/human/review_log.jsonl`.

---

## OCR Strategy

### Short term (Phase 1.5)

Use cloud vision APIs (Claude/Gemini) + baseline OCR as labeling aids.

### Long term (Phase 3)

Treat production OCR engine as a separate evaluation. Preserve enough metadata and reviewed plate crops to compare:

- EasyOCR baseline
- Custom OCR model
- LPR-style model on Jetson

### Plate metadata to preserve

- `plate_layout_type` (one_line / two_line)
- `line_count`
- `governorate_text_visible`
- `arabic_visible`
- `latin_visible`
- `plate_color_style`

---

## Configuration

| Parameter | Default | Description |
| --- | --- | --- |
| `--camera` | `0` | Camera device index |
| `--vehicle-model` | `yolo26x.pt` | Vehicle detector model |
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
| `--save-better-only` | `true` | Save again only if new crop is better |
| `--save-hard-negatives` | `true` | Keep useful negative examples |
| `--no-display` | `false` | Run headless (no OpenCV window) |
| `--session-id` | auto-generated | Session identifier |

---

## Dependencies

### Phase 1 — Collection

```text
ultralytics          # YOLO26x
opencv-python        # Camera capture + display
numpy                # Array operations
```

### Phase 1.5 — Review

```text
anthropic            # Claude Vision API
google-generativeai  # Gemini Vision API (alternative)
Pillow               # Image loading for API
```

---

## Final Deployment Target

Police-car-mounted Jetson Orin Nano (8GB):

- Multiple simultaneous camera feeds
- Single fine-tuned YOLO26m detecting vehicle + plate
- Lightweight vehicle brand/model classifier (ResNet-18 or EfficientNet-B0)
- Real-time OCR for Arabic + Latin Syrian plates
- Reliable vehicle-plate association
- Optimized edge inference (TensorRT / DeepStream)
- Output per detection: **vehicle photo + brand/model + plate photo + plate text**

---

## Phased Roadmap

| Phase | Hardware | What | Models / Stack |
| --- | --- | --- | --- |
| **1 - Collect** | Mac | Real-time data collection from USB webcam | YOLO26x (vehicle) + plate detector (two models) |
| **1.5 - AI Review** | Cloud API | Label triage: bbox verify, brand detect, plate OCR | Claude / Gemini Vision API |
| **1.5b - Human Review** | Mac | Approve, reject, or correct AI labels | Python review CLI |
| **2 - Train** | Mac / Cloud | Train on approved data only | YOLO26m (vehicle+plate) + brand classifier + OCR model |
| **3 - Deploy** | Orin Nano | Edge inference in police car, multi-cam | TensorRT YOLO26m + classifier + OCR |

---

## Phase 2 Training Notes

- **YOLO fine-tuning:** Use only approved labels from Phase 1.5b. Fine-tune YOLO26m (pre-trained on COCO) to detect both `vehicle` and `plate` in a single pass.
- **Brand classifier:** Use AI-generated + human-verified brand labels from vehicle crops. Train ResNet-18 or EfficientNet-B0. Target: top Syrian car brands (20-50 classes).
- **OCR training:** Use AI-generated + human-verified plate text labels. Fine-tune or train custom OCR for Arabic + Latin Syrian plate format. Compare against EasyOCR baseline.

---

## Error Handling

- Camera not found: print error and exit
- Model download failure: print error and exit
- Disk full: catch IOError on save, warn and continue
- No detections: continue showing live feed, no saves
- API rate limit (Phase 1.5): retry with exponential backoff
- API error (Phase 1.5): log error, skip image, continue

---

## Out of Scope (Phase 1)

- Final OCR deployment (Phase 3)
- Final vehicle brand/model classifier training (Phase 2)
- Database storage
- Web interface
- Production multi-camera synchronization (Phase 3)
- Full DeepStream implementation (Phase 3)
- Model fine-tuning scripts (Phase 2)
