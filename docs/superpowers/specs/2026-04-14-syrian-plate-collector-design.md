# Syrian License Plate Data Collector — Design Spec

## Purpose

A Python-based data collection pipeline that uses a USB webcam to detect cars and license plates in real-time, saving high-quality training data for:

1. Fine-tuning a single YOLO26 model to detect both cars and plates
2. Collecting cropped plate images for future Arabic/Latin OCR training
3. Collecting cropped car images for car brand/model classifier training

Target deployment: Jetson Orin Nano (8GB) in police cars — Phase 1 runs on Mac for data collection.

## Scope

- Syrian license plates (Arabic + Latin characters)
- Live USB webcam feed with real-time display
- Smart deduplication (avoid saving thousands of identical frames)
- Training-quality image output (full-res for crops, high-quality JPEG for frames)
- Car-plate association (link each plate to its parent car)

## Architecture

```text
USB Webcam
    │
    ▼
OpenCV Capture (full resolution)
    │
    ├──▶ YOLO26x ── Car Detection (COCO pre-trained, conf 0.3)
    │
    ├──▶ Plate Detection Model ── License Plate Detection (full frame, conf 0.3)
    │
    ▼
Association Engine ── Link plates to cars (plate bbox inside car bbox)
    │
    ├──▶ Live Display (bounding boxes + counters)
    │
    ├──▶ Dedup Check ── Only save when new/changed object detected
    │       │
    │       ├──▶ Full Frame Saver ── dataset/images/ (JPEG 95%) + dataset/labels/ (pseudo-labels)
    │       ├──▶ Car Cropper ── crops/cars/YYYY-MM-DD/ (PNG)
    │       ├──▶ Plate Cropper ── crops/plates/YYYY-MM-DD/ (PNG)
    │       └──▶ Detection Log ── detections.jsonl (car-plate associations per save)
    │
    └──▶ Stats Tracker ── stats.json
```

## Components

### 1. Camera Capture

- OpenCV VideoCapture from USB webcam (device index 0)
- Capture at full camera resolution (no downscaling for saved images)
- Detection runs on a resized frame for speed; crops taken from original resolution frame

### 2. Car Detection

- Model: YOLO26x pre-trained on COCO
- Classes: `car`, `truck`, `bus` (vehicle types)
- Confidence threshold: 0.3 (lower for data collection — capture borderline cases)
- Runs on full frame

### 3. Plate Detection

- Model: Pre-trained license plate detection model (YOLO-format, sourced from Roboflow/HuggingFace — specific model to be selected during implementation)
- Runs on **full frame** in parallel with car detection (not sequential, not on car crops)
- Confidence threshold: 0.3

### 4. Car-Plate Association

- After both models run, link plates to cars using spatial overlap
- A plate belongs to a car if the plate bounding box is **inside or mostly overlapping** (>50% IoU) the car bounding box
- Unmatched plates (no parent car) are still saved — the car body may be partially out of frame
- Association saved in `detections.jsonl` per save event

### 5. Smart Deduplication

- Track detected objects using **centroid tracking** (center point distance between frames)
- Assign a tracker ID to each unique car/plate
- Save only when:
  - A new object appears (centroid not within distance threshold of any known object)
  - An existing tracked object hasn't been saved for `cooldown` seconds
- Cooldown: minimum 3 seconds between saves of the same tracked object
- Remove stale trackers after 5 seconds of no match (object left the frame)
- **Full frames are only saved when a crop save is triggered** — no standalone frame saves
- Uses simple centroid tracking (no deep SORT needed for Phase 1)

### 6. Image Saving

#### Full Frames + Pseudo-Labels (for fine-tuning)

- Save full-resolution frame as **JPEG 95% quality**: `dataset/images/frame_{counter:06d}.jpg`
- Save corresponding YOLO pseudo-label: `dataset/labels/frame_{counter:06d}.txt`
- Label format: `class_id x_center y_center width height` (normalized 0-1)
- Class mapping: `0 = car`, `1 = plate`
- **Only saved when dedup triggers a crop save** — not on every detection frame
- **Important:** These are auto-generated pseudo-labels, NOT verified ground truth. They require review/correction before use in training.

#### Car Crops (for brand classifier training)

- Crop car region from full-res frame with 10% padding
- Save as **PNG**: `crops/cars/YYYY-MM-DD/car_{counter:04d}_{timestamp}.png`
- These crops will be labeled with brand/model in Phase 1.5

#### Plate Crops (for OCR training)

- Crop plate region from full-res frame with 5% padding
- Save as **PNG**: `crops/plates/YYYY-MM-DD/plate_{counter:04d}_{timestamp}.png`

#### Detection Log (car-plate associations)

- Append one JSON line per save event to `detections.jsonl`
- Links car crops to their plate crops and to the full frame

```json
{
  "timestamp": "2026-04-14T14:30:22",
  "frame": "dataset/images/frame_000001.jpg",
  "detections": [
    {
      "car_crop": "crops/cars/2026-04-14/car_0001_20260414_143022.png",
      "car_bbox": [120, 80, 450, 320],
      "plate_crop": "crops/plates/2026-04-14/plate_0001_20260414_143022.png",
      "plate_bbox": [200, 280, 340, 310]
    },
    {
      "car_crop": "crops/cars/2026-04-14/car_0002_20260414_143022.png",
      "car_bbox": [500, 100, 780, 350],
      "plate_crop": null,
      "plate_bbox": null
    }
  ]
}
```

### 7. Live Display

- OpenCV window showing the camera feed
- Green bounding boxes around cars with label "CAR"
- Red bounding boxes around plates with label "PLATE"
- Lines connecting plates to their parent cars
- Overlay counters: "Cars: X | Plates: X | Frames: X"
- Press `q` to quit cleanly

## Output Directory Structure

```text
output/
├── dataset/
│   ├── images/
│   │   ├── frame_000001.jpg
│   │   └── frame_000002.jpg
│   ├── labels/
│   │   ├── frame_000001.txt
│   │   └── frame_000002.txt
│   └── classes.txt              # "car\nplate"
├── crops/
│   ├── cars/
│   │   └── 2026-04-14/
│   │       ├── car_0001_20260414_143022.png
│   │       └── car_0002_20260414_143045.png
│   └── plates/
│       └── 2026-04-14/
│           ├── plate_0001_20260414_143022.png
│           └── plate_0002_20260414_143045.png
├── detections.jsonl             # Car-plate associations per save
└── stats.json                   # Session stats (total counts, duration)
```

## Dependencies

### Phase 1 — Data Collection

```text
ultralytics          # YOLO26x
opencv-python        # Camera capture + display
numpy                # Array operations
```

### Phase 1.5 — AI Review

```text
anthropic            # Claude Vision API (or google-generativeai for Gemini)
Pillow               # Image loading for API
```

## Configuration

All configurable via command-line arguments or a config dict at the top of the script:

| Parameter | Default | Description |
| --- | --- | --- |
| `--camera` | 0 | Camera device index |
| `--car-conf` | 0.3 | Car detection confidence threshold |
| `--plate-conf` | 0.3 | Plate detection confidence threshold |
| `--output` | `./output` | Output directory |
| `--cooldown` | 3 | Seconds between saves of same object |
| `--no-display` | false | Run headless (no OpenCV window) |

## Final Deployment Target

Police car mounted system on Jetson Orin Nano (8GB):

- Multiple simultaneous camera feeds
- Single fine-tuned YOLO26m detecting cars + plates
- Lightweight brand/model classifier (ResNet/EfficientNet) on car crops
- Real-time OCR (Arabic + Latin) on detected plates
- Saves: car photo, car brand/model, plate photo, plate text
- Car-plate associations maintained
- Must run efficiently on edge hardware

## Phased Roadmap

| Phase | Hardware | What | Models |
| --- | --- | --- | --- |
| **1 - Collect** | Mac | This tool — collect training data from single webcam | YOLO26x + plate model (two models, temporary) |
| **1.5 - AI Review** | Cloud API | Send images to Claude/Gemini Vision for auto-verification and labeling | Claude Vision / Gemini Vision API |
| **1.5b - Human Review** | Mac | Human reviews AI-flagged items, approves/rejects via review tool | Python review CLI |
| **2 - Train** | Cloud/Mac | Fine-tune models on approved data only | YOLO26m + ResNet/EfficientNet classifier |
| **3 - Deploy** | Orin Nano | Multi-cam, detection + brand ID + OCR, police car | YOLO26m + brand classifier + EasyOCR |

## Phase 1.5 — AI-Assisted Label Verification

Three **separate, independent** Python modules — each callable on demand on any image. Run one, all, or re-run individually without affecting the others.

### Module 1: `verify_bbox(image_path)`

Verify bounding box accuracy on a full frame + its pseudo-label.

- Input: full frame image + YOLO label file
- Sends to Claude/Gemini Vision API
- Output: appends to `review/bbox_results.jsonl`

```json
{
  "image": "frame_000001.jpg",
  "bbox_quality": "good",
  "issues": [],
  "certainty": "high"
}
```

### Module 2: `detect_brand(car_crop_path)`

Identify car brand and model from a car crop image.

- Input: single car crop image
- Sends to Claude/Gemini Vision API
- Output: appends to `review/brand_results.jsonl`

```json
{
  "image": "car_0001_20260414_143022.png",
  "brand": "Toyota",
  "model": "Corolla",
  "year_estimate": "2015-2020",
  "certainty": "high"
}
```

### Module 3: `read_plate(plate_crop_path)`

OCR a plate crop image for Arabic + Latin text.

- Input: single plate crop image
- Sends to Claude/Gemini Vision API
- Output: appends to `review/plate_results.jsonl`

```json
{
  "image": "plate_0001_20260414_143022.png",
  "plate_text_original": "دمشق ٣٤٥٦٧٨",
  "plate_text_latin": "Damascus 345678",
  "certainty": "medium"
}
```

### Certainty Levels

- **high** — AI is confident, auto-approved (no human review needed)
- **medium** — AI is somewhat sure, flagged for human review
- **low** — AI is unsure or confused, requires human review

### CLI Usage

```bash
# Run individually on a single image
python review.py bbox --image dataset/images/frame_000001.jpg
python review.py brand --image crops/cars/2026-04-14/car_0001.png
python review.py plate --image crops/plates/2026-04-14/plate_0001.png

# Run on entire folder
python review.py bbox --dir dataset/images/
python review.py brand --dir crops/cars/
python review.py plate --dir crops/plates/

# Run all three on everything
python review.py all --output-dir output/
```

## Phase 1.5b — Human Review Tool

A Python CLI tool that shows flagged items one by one for human approval.

### How It Works

```bash
python human_review.py --review-dir review/ --crops-dir crops/
```

For each flagged item (certainty = medium or low):

1. Opens the image in a viewer (OpenCV window or system image viewer)
2. Shows the AI label (brand, plate text, or bbox quality)
3. Human presses:
   - `y` — approve as-is
   - `n` — reject (moves to rejected/)
   - `e` — edit (type corrected label)
   - `s` — skip for later

### Review Output

```text
output/
├── dataset/
│   ├── approved/
│   │   ├── images/              # Verified frames
│   │   ├── labels/              # Corrected YOLO labels
│   │   ├── car_labels.jsonl     # Brand/model labels (verified)
│   │   └── plate_labels.jsonl   # OCR text labels (verified)
│   └── rejected/
│       ├── images/              # Bad detections
│       └── labels/              # Incorrect labels
├── review/
│   ├── bbox_results.jsonl       # AI bbox verification results
│   ├── brand_results.jsonl      # AI brand detection results
│   ├── plate_results.jsonl      # AI plate OCR results
│   └── human_review_log.jsonl   # Human decisions (approve/reject/edit)
```

## Phase 2 Training Notes

- **YOLO fine-tuning:** Use only approved labels from Phase 1.5b. Fine-tune YOLO26m (pre-trained on COCO) to detect both `car` and `plate` in a single pass.
- **Brand classifier:** Use AI-generated + human-verified brand labels from car crops. Train a lightweight classifier (ResNet-18 or EfficientNet-B0). Target: top Syrian car brands (likely 20-50 classes).
- **OCR training:** Use AI-generated + human-verified plate text labels. Fine-tune EasyOCR or train a custom model for Arabic + Latin Syrian plate format.

## Error Handling

- Camera not found: print error and exit
- Model download failure: print error and exit
- Disk full: catch IOError on save, warn and continue
- No detections: continue showing live feed, no saves
- API rate limit (Phase 1.5): retry with exponential backoff
- API error (Phase 1.5): log error, skip image, continue

## Out of Scope (Phase 1)

- OCR text reading (Phase 1.5 / Phase 3)
- Car brand/model classification (Phase 1.5 / Phase 3)
- Database storage
- Web interface
- Multi-camera support (Phase 3)
- Deep SORT or Re-ID tracking
- Model training/fine-tuning scripts (Phase 2)
