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
