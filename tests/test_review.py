import numpy as np
from pathlib import Path
from src.review.grid import compose_grid, parse_grid_size


def test_parse_grid_size():
    assert parse_grid_size("3x3") == (3, 3)
    assert parse_grid_size("2x2") == (2, 2)
    assert parse_grid_size("4x4") == (4, 4)


def test_compose_grid_creates_image():
    images = [np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8) for _ in range(4)]
    paths = [Path(f"img_{i}.png") for i in range(4)]
    composite, mapping = compose_grid(images, paths, grid_size=(2, 2), tile_px=256)
    assert composite.shape == (512, 512, 3)
    assert len(mapping) == 4
    assert mapping[0] == paths[0]


def test_compose_grid_partial_batch():
    images = [np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8) for _ in range(3)]
    paths = [Path(f"img_{i}.png") for i in range(3)]
    composite, mapping = compose_grid(images, paths, grid_size=(2, 2), tile_px=256)
    assert composite.shape == (512, 512, 3)
    assert len(mapping) == 3
