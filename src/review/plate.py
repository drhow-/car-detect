# src/review/plate.py
"""Module 3: OCR plate crops for Arabic + Latin text using Vision API."""
import json
import re
import base64
from pathlib import Path

import cv2

from src.review.grid import compose_grid, parse_grid_size

_PLATE_PATTERNS = [
    re.compile(r'^[\u0600-\u06FF]{2,8}\s?\d{3,7}$'),
    re.compile(r'^[A-Z]{2,8}\s?\d{3,7}$'),
    re.compile(r'^[\d\u0660-\u0669]{4,8}$'),
    re.compile(r'^\d{4,8}$'),
    re.compile(r'^[\u0600-\u06FF]+$'),
]


def _validate_plate_format(text):
    if not text or not text.strip():
        return False, "unreadable"

    cleaned = text.strip()
    cleaned = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    for pattern in _PLATE_PATTERNS:
        if pattern.match(cleaned):
            return True, None

    lines = cleaned.split('\n')
    if len(lines) >= 2:
        has_digit_line = any(re.match(r'^[\d\u0660-\u0669]{3,}$', l.strip()) for l in lines)
        if has_digit_line:
            return True, None

    return False, "hallucination_suspected"


def _encode_image(img):
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.standard_b64encode(buf).decode("utf-8")


def _build_prompt(n_cells):
    return f"""You are reading Syrian license plates from cropped photos.

This image is a grid of {n_cells} plate crops. Each cell has an index number (0-{n_cells-1}) in the top-left.

Syrian plates typically have Arabic governorate text (e.g. دمشق, حلب) and digits. Some have Latin transliterations.

For each cell, read the plate and return a JSON array of length {n_cells} in row-major order.

Each element must have:
- "cell_index": integer
- "plate_text_original": string (Arabic text as-is)
- "plate_text_latin": string (Latin transliteration/translation)
- "plate_layout_type": "one_line" | "two_line"
- "line_count": integer
- "governorate_text_visible": boolean
- "arabic_visible": boolean
- "latin_visible": boolean
- "plate_color_style": string (e.g. "white_blue", "white_red", "yellow")
- "certainty": "high" | "medium" | "low"

If a cell is black/empty or illegible, return with certainty "low" and plate_text_original "".

Return ONLY the JSON array, no other text."""


def read_plate_batch(crop_paths, client, model="claude-sonnet-4-20250514",
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
        max_tokens=3000,
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
        return [{"image": paths[i].name, "plate_text_original": "", "plate_text_latin": "",
                 "certainty": "low", "format_valid": False, "format_flag": "parse_failed",
                 "error": "parse_failed"}
                for i in range(len(paths))]

    output = []
    for item in results:
        idx = item.get("cell_index", -1)
        if idx in mapping:
            item["image"] = mapping[idx].name

        is_valid, flag = _validate_plate_format(item.get("plate_text_original", ""))
        item["format_valid"] = is_valid
        item["format_flag"] = flag

        if not is_valid:
            item["certainty"] = "low"

        output.append(item)

    return output
