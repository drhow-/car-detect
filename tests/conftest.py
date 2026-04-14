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
