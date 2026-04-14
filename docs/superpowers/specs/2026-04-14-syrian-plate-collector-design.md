# Syrian License Plate Data Collector — Design Spec

## Purpose

A Python-based data collection pipeline that uses a USB webcam to detect cars and license plates in real-time, saving high-quality training data for:
1. Fine-tuning a single YOLO26 model to detect both cars and plates
2. Collecting cropped plate images for future Arabic/Latin OCR training

Target deployment: Jetson Orin Nano (8GB) — but Phase 1 runs on Mac for data collection.

## Scope

- Syrian license plates (Arabic + Latin characters)
- Live USB webcam feed with real-time display
- Smart deduplication (avoid saving thousands of identical frames)
- Training-quality image output (full-res PNG)

## Architecture

```
USB Webcam
    │
    ▼
OpenCV Capture (full resolution)
    │
    ▼
YOLO26x ── Car Detection (COCO pre-trained)
    │
    ▼
Plate Detection Model ── License Plate Detection (open-source pre-trained)
    │
    ├──▶ Live Display (OpenCV window with bounding boxes + counters)
    │
    ├──▶ Full Frame Saver ── dataset/images/ + dataset/labels/ (YOLO format)
    │
    ├──▶ Car Cropper ── crops/cars/YYYY-MM-DD/
    │
    └──▶ Plate Cropper ── crops/plates/YYYY-MM-DD/
```

## Components

### 1. Camera Capture
- OpenCV VideoCapture from USB webcam (device index 0)
- Capture at full camera resolution (no downscaling for saved images)
- Detection runs on a resized frame for speed; crops taken from original resolution frame

### 2. Car Detection
- Model: YOLO26x pre-trained on COCO
- Classes: `car`, `truck`, `bus` (vehicle types)
- Confidence threshold: 0.5 (configurable)

### 3. Plate Detection
- Model: Pre-trained license plate detection model (YOLO-format, sourced from Roboflow/HuggingFace)
- Runs on detected car regions (two-stage) OR full frame (single-stage) — depending on available model
- Confidence threshold: 0.5 (configurable)

### 4. Smart Deduplication
- Track detected objects using IoU (Intersection over Union) between frames
- Assign a tracker ID to each unique car/plate
- Save only when:
  - A new object appears (not tracked before)
  - An existing object changes significantly (IoU < 0.5 with previous position)
- Cooldown: minimum 3 seconds between saves of the same tracked object
- Uses simple centroid/IoU tracking (no deep SORT needed for Phase 1)

### 5. Image Saving

#### Full Frames + YOLO Labels (for fine-tuning)
- Save full-resolution frame as PNG: `dataset/images/frame_{counter:06d}.png`
- Save corresponding YOLO label: `dataset/labels/frame_{counter:06d}.txt`
- Label format: `class_id x_center y_center width height` (normalized 0-1)
- Class mapping: `0 = car`, `1 = plate`
- Only save frames that contain at least one detection

#### Car Crops (for review/QA)
- Crop car region from full-res frame with 10% padding
- Save as PNG: `crops/cars/YYYY-MM-DD/car_{counter:04d}_{timestamp}.png`

#### Plate Crops (for OCR training)
- Crop plate region from full-res frame with 5% padding
- Save as PNG: `crops/plates/YYYY-MM-DD/plate_{counter:04d}_{timestamp}.png`

### 6. Live Display
- OpenCV window showing the camera feed
- Green bounding boxes around cars with label "CAR"
- Red bounding boxes around plates with label "PLATE"
- Overlay counters: "Cars: X | Plates: X | Frames: X"
- Press `q` to quit cleanly

## Output Directory Structure

```
output/
├── dataset/
│   ├── images/
│   │   ├── frame_000001.png
│   │   └── frame_000002.png
│   ├── labels/
│   │   ├── frame_000001.txt
│   │   └── frame_000002.txt
│   └── classes.txt          # "car\nplate"
├── crops/
│   ├── cars/
│   │   └── 2026-04-14/
│   │       ├── car_0001_20260414_143022.png
│   │       └── car_0002_20260414_143045.png
│   └── plates/
│       └── 2026-04-14/
│           ├── plate_0001_20260414_143022.png
│           └── plate_0002_20260414_143045.png
└── stats.json               # Session stats (total counts, duration)
```

## Dependencies

```
ultralytics          # YOLO26x
opencv-python        # Camera capture + display
numpy                # Array operations
```

## Configuration

All configurable via command-line arguments or a config dict at the top of the script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--camera` | 0 | Camera device index |
| `--car-conf` | 0.5 | Car detection confidence threshold |
| `--plate-conf` | 0.5 | Plate detection confidence threshold |
| `--output` | `./output` | Output directory |
| `--cooldown` | 3 | Seconds between saves of same object |
| `--no-display` | false | Run headless (no OpenCV window) |

## Final Deployment Target

Police car mounted system on Jetson Orin Nano (8GB):

- Multiple simultaneous camera feeds
- Single fine-tuned YOLO26m detecting cars + plates
- Real-time OCR (Arabic + Latin) on detected plates
- Saves: car photo, plate photo, plate text
- Must run efficiently on edge hardware

## Phased Roadmap

| Phase | Hardware | What | Model |
| ----- | -------- | ---- | ----- |
| **1 - Collect** | Mac | This tool — collect training data from single webcam | YOLO26x + plate model (two models) |
| **2 - Train** | Cloud/Mac | Fine-tune single model for car+plate | YOLO26m |
| **3 - Deploy** | Orin Nano | Multi-cam, single model + OCR, police car | YOLO26m (fine-tuned) + EasyOCR |

## Error Handling

- Camera not found: print error and exit
- Model download failure: print error and exit
- Disk full: catch IOError on save, warn and continue
- No detections: continue showing live feed, no saves

## Out of Scope (Phase 1)

- OCR text reading
- Database storage
- Web interface
- Multi-camera support (Phase 3)
- Deep SORT or Re-ID tracking
- Model training/fine-tuning scripts (Phase 2)
