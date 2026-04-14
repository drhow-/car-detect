# src/review/brand.py
"""Module 2: Detect vehicle brand/model using Vision API."""
import json
import base64
from pathlib import Path

import cv2

from src.review.grid import compose_grid, parse_grid_size


def _encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode("utf-8")


def _build_prompt(n_cells):
    return f"""You are identifying vehicle brands and models from cropped photos.

This image is a grid of {n_cells} vehicle crops. Each cell has an index number (0-{n_cells-1}) in the top-left.

For each cell, identify the vehicle and return a JSON array of length {n_cells} in row-major order.

Each element must have:
- "cell_index": integer
- "brand": string (e.g. "Toyota", "Hyundai", "Kia")
- "model": string (e.g. "Corolla", "Accent")
- "year_estimate": string range (e.g. "2015-2020") or "unknown"
- "certainty": "high" | "medium" | "low"

If a cell is black/empty or unrecognizable, return {{"cell_index": N, "brand": "unknown", "model": "unknown", "year_estimate": "unknown", "certainty": "low"}}

Return ONLY the JSON array, no other text."""


def detect_brand_batch(crop_paths, client, model="claude-sonnet-4-20250514",
                       grid_size="3x3", tile_px=512):
    rows, cols = parse_grid_size(grid_size)
    max_cells = rows * cols

    images = []
    paths = []
    for cp in crop_paths[:max_cells]:
        cp = Path(cp)
        img = cv2.imread(str(cp))
        if img is not None:
            images.append(img)
            paths.append(cp)

    if not images:
        return []

    composite, mapping = compose_grid(images, paths, parse_grid_size(grid_size), tile_px)
    img_b64 = _encode_image(composite)

    response = client.messages.create(
        model=model,
        max_tokens=2000,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64}},
                {"type": "text", "text": _build_prompt(len(mapping))},
            ],
        }],
    )

    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]

    try:
        results = json.loads(text)
    except json.JSONDecodeError:
        return [{"image": paths[i].name, "brand": "unknown", "model": "unknown",
                 "year_estimate": "unknown", "certainty": "low", "error": "parse_failed"}
                for i in range(len(paths))]

    output = []
    for item in results:
        idx = item.get("cell_index", -1)
        if idx in mapping:
            item["image"] = mapping[idx].name
        output.append(item)

    return output
