# src/review/bbox.py
"""Module 1: Verify bounding box accuracy using Vision API."""
import json
import base64
from pathlib import Path

import cv2

from src.review.grid import compose_grid, parse_grid_size


def _encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode("utf-8")


def _build_prompt(n_cells):
    return f"""You are reviewing vehicle and license plate detection bounding boxes.

This image is a grid of {n_cells} frames. Each cell has an index number (0-{n_cells-1}) in the top-left.
The original frames had bounding boxes drawn on them before being placed in the grid.

For each cell, evaluate the bounding box quality and return a JSON array of length {n_cells} in row-major order.

Each element must have:
- "cell_index": integer
- "bbox_quality": "good" | "acceptable" | "poor"
- "missing_objects": list of descriptions of vehicles/plates visible but not boxed
- "false_positives": list of descriptions of boxes around non-vehicle/non-plate objects
- "certainty": "high" | "medium" | "low"

If a cell is black/empty, return {{"cell_index": N, "bbox_quality": "good", "missing_objects": [], "false_positives": [], "certainty": "low"}}

Return ONLY the JSON array, no other text."""


def verify_bbox_batch(frame_paths, label_dir, client, model="claude-sonnet-4-20250514",
                      grid_size="2x2", tile_px=512):
    rows, cols = parse_grid_size(grid_size)
    max_cells = rows * cols

    images = []
    paths = []
    for fp in frame_paths[:max_cells]:
        fp = Path(fp)
        img = cv2.imread(str(fp))
        if img is None:
            continue

        label_path = Path(label_dir) / fp.with_suffix(".txt").name
        if label_path.exists():
            h, w = img.shape[:2]
            for line in label_path.read_text().strip().split("\n"):
                parts = line.strip().split()
                if len(parts) == 5:
                    cls = int(parts[0])
                    xc, yc, bw, bh = [float(x) for x in parts[1:]]
                    x1 = int((xc - bw / 2) * w)
                    y1 = int((yc - bh / 2) * h)
                    x2 = int((xc + bw / 2) * w)
                    y2 = int((yc + bh / 2) * h)
                    color = (0, 255, 0) if cls == 0 else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        images.append(img)
        paths.append(fp)

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
        return [{"image": paths[i].name, "bbox_quality": "unknown", "certainty": "low",
                 "missing_objects": [], "false_positives": [], "error": "parse_failed"}
                for i in range(len(paths))]

    output = []
    for item in results:
        idx = item.get("cell_index", -1)
        if idx in mapping:
            item["image"] = mapping[idx].name
        output.append(item)

    return output
