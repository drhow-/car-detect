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

## Architecture

```text
USB Webcam
    │
    ▼
OpenCV Capture (full resolution)
    │
    ▼
YOLO26x ── Car Detection (COCO pre-trained, conf 0.3)
    │
    ▼
Plate Detection Model ── License Plate Detection (full frame, conf 0.3)
    │
    ├──▶ Live Display (OpenCV window with bounding boxes + counters)
    │
    ├──▶ Full Frame Saver ── dataset/images/ (JPEG 95%) + dataset/labels/ (pseudo-labels)
    │
    ├──▶ Car Cropper ── crops/cars/YYYY-MM-DD/ (PNG, for brand classifier training)
    │
    └──▶ Plate Cropper ── crops/plates/YYYY-MM-DD/ (PNG, for OCR training)
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

### 3. Plate Detection

- Model: Pre-trained license plate detection model (YOLO-format, sourced from Roboflow/HuggingFace — specific model to be selected during implementation)
- Runs on **full frame** (not just car crops — plates can appear without fully visible car body)
- Confidence threshold: 0.3

### 4. Smart Deduplication

- Track detected objects using **centroid tracking** (center point distance between frames)
- Assign a tracker ID to each unique car/plate
- Save only when:
  - A new object appears (centroid not within distance threshold of any known object)
  - An existing tracked object hasn't been saved for `cooldown` seconds
- Cooldown: minimum 3 seconds between saves of the same tracked object
- Remove stale trackers after 5 seconds of no match (object left the frame)
- Uses simple centroid tracking (no deep SORT needed for Phase 1)

### 5. Image Saving

#### Full Frames + Pseudo-Labels (for fine-tuning)

- Save full-resolution frame as **JPEG 95% quality**: `dataset/images/frame_{counter:06d}.jpg`
- Save corresponding YOLO pseudo-label: `dataset/labels/frame_{counter:06d}.txt`
- Label format: `class_id x_center y_center width height` (normalized 0-1)
- Class mapping: `0 = car`, `1 = plate`
- Only save frames that contain at least one detection
- **Important:** These are auto-generated pseudo-labels, NOT verified ground truth. They require manual review/correction before use in training (Phase 2).

#### Car Crops (for brand classifier training)

- Crop car region from full-res frame with 10% padding
- Save as **PNG**: `crops/cars/YYYY-MM-DD/car_{counter:04d}_{timestamp}.png`
- These crops will be labeled with brand/model in Phase 2

#### Plate Crops (for OCR training)

- Crop plate region from full-res frame with 5% padding
- Save as **PNG**: `crops/plates/YYYY-MM-DD/plate_{counter:04d}_{timestamp}.png`

### 6. Live Display

- OpenCV window showing the camera feed
- Green bounding boxes around cars with label "CAR"
- Red bounding boxes around plates with label "PLATE"
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
└── stats.json                   # Session stats (total counts, duration)
```

## Dependencies

```text
ultralytics          # YOLO26x
opencv-python        # Camera capture + display
numpy                # Array operations
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
- Must run efficiently on edge hardware

## Phased Roadmap

| Phase | Hardware | What | Models |
| --- | --- | --- | --- |
| **1 - Collect** | Mac | This tool — collect training data from single webcam | YOLO26x + plate model (two models, temporary) |
| **2 - Train** | Cloud/Mac | Fine-tune single YOLO for car+plate, train brand classifier, review pseudo-labels | YOLO26m + ResNet/EfficientNet classifier |
| **3 - Deploy** | Orin Nano | Multi-cam, detection + brand ID + OCR, police car | YOLO26m + brand classifier + EasyOCR |

## Phase 2 Training Notes

- **YOLO fine-tuning:** Use reviewed/corrected pseudo-labels from Phase 1 dataset. Fine-tune YOLO26m (pre-trained on COCO) to detect both `car` and `plate` in a single pass.
- **Brand classifier:** Label car crops with brand/model names. Train a lightweight classifier (ResNet-18 or EfficientNet-B0) on labeled car crops. Target: top Syrian car brands (likely 20-50 classes).
- **OCR training:** Label plate crops with plate text. Fine-tune EasyOCR or train a custom model for Arabic + Latin Syrian plate format.

## Error Handling

- Camera not found: print error and exit
- Model download failure: print error and exit
- Disk full: catch IOError on save, warn and continue
- No detections: continue showing live feed, no saves

## Out of Scope (Phase 1)

- OCR text reading (Phase 3)
- Car brand/model classification (Phase 2/3)
- Database storage
- Web interface
- Multi-camera support (Phase 3)
- Deep SORT or Re-ID tracking
- Model training/fine-tuning scripts (Phase 2)
