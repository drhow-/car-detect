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
