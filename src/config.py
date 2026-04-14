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
