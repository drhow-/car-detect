import cv2
import numpy as np
from pathlib import Path


def parse_grid_size(s):
    parts = s.lower().split("x")
    return int(parts[0]), int(parts[1])


def _resize_with_padding(img, tile_px):
    h, w = img.shape[:2]
    scale = min(tile_px / w, tile_px / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h))
    canvas = np.zeros((tile_px, tile_px, 3), dtype=np.uint8)
    y_off = (tile_px - new_h) // 2
    x_off = (tile_px - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def compose_grid(images, paths, grid_size=(3, 3), tile_px=512):
    rows, cols = grid_size
    canvas = np.zeros((rows * tile_px, cols * tile_px, 3), dtype=np.uint8)
    mapping = {}

    for i, (img, path) in enumerate(zip(images, paths)):
        if i >= rows * cols:
            break
        r, c = divmod(i, cols)
        tile = _resize_with_padding(img, tile_px)
        cv2.putText(tile, str(i), (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y1, y2 = r * tile_px, (r + 1) * tile_px
        x1, x2 = c * tile_px, (c + 1) * tile_px
        canvas[y1:y2, x1:x2] = tile
        mapping[i] = path

    return canvas, mapping
