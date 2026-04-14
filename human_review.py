# human_review.py
"""Human review tool for AI-flagged items."""
import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2


def _load_review_queue(review_dir, module):
    results_file = Path(review_dir) / "ai" / f"{module}_results.jsonl"
    items = []
    if results_file.exists():
        for line in results_file.read_text().strip().split("\n"):
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("certainty") in ("medium", "low"):
                items.append((module, item))

    if module == "plate":
        suspect_file = Path(review_dir) / "queues" / "plate_format_suspect.jsonl"
        if suspect_file.exists():
            for line in suspect_file.read_text().strip().split("\n"):
                if not line.strip():
                    continue
                items.append(("plate_suspect", json.loads(line)))

    return items


def _find_image(image_name, raw_dir):
    for p in Path(raw_dir).rglob(image_name):
        return p
    return None


def _display_info(module, item):
    print("\n" + "=" * 60)
    print(f"Module: {module} | Image: {item.get('image', '?')}")
    print(f"Certainty: {item.get('certainty', '?')}")

    if module in ("brand",):
        print(f"Brand: {item.get('brand', '?')} | Model: {item.get('model', '?')}")
        print(f"Year: {item.get('year_estimate', '?')}")
    elif module in ("plate", "plate_suspect"):
        print(f"Text (original): {item.get('plate_text_original', '?')}")
        print(f"Text (latin): {item.get('plate_text_latin', '?')}")
        print(f"Layout: {item.get('plate_layout_type', '?')}")
        if item.get("format_flag"):
            print(f"FORMAT FLAG: {item['format_flag']}")
    elif module == "bbox":
        print(f"BBox quality: {item.get('bbox_quality', '?')}")
        if item.get("missing_objects"):
            print(f"Missing: {item['missing_objects']}")
        if item.get("false_positives"):
            print(f"False positives: {item['false_positives']}")

    print("-" * 60)
    print("[y] approve | [n] reject | [e] edit | [s] skip")


def _promote(module, item, raw_dir, output_dir, correction=None):
    image_name = item.get("image", "")
    src = _find_image(image_name, raw_dir)
    if not src:
        print(f"  WARNING: source image not found: {image_name}")
        return

    if module == "bbox":
        dest_dir = Path(output_dir) / "approved" / "detector" / "images"
    elif module in ("plate", "plate_suspect"):
        dest_dir = Path(output_dir) / "approved" / "ocr" / "plate_crops"
    elif module == "brand":
        dest_dir = Path(output_dir) / "approved" / "vehicle_classifier" / "vehicle_crops"
    else:
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)

    if module in ("plate", "plate_suspect"):
        label_file = Path(output_dir) / "approved" / "ocr" / "plate_labels.jsonl"
        label_data = {k: v for k, v in item.items()}
        if correction:
            label_data["plate_text_original"] = correction
            label_data["human_corrected"] = True
        label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(label_file, "a") as f:
            f.write(json.dumps(label_data, ensure_ascii=False) + "\n")
    elif module == "brand":
        label_file = Path(output_dir) / "approved" / "vehicle_classifier" / "brand_labels.jsonl"
        label_data = {k: v for k, v in item.items()}
        if correction:
            label_data["brand"] = correction
            label_data["human_corrected"] = True
        label_file.parent.mkdir(parents=True, exist_ok=True)
        with open(label_file, "a") as f:
            f.write(json.dumps(label_data, ensure_ascii=False) + "\n")


def _reject(module, item, raw_dir, output_dir):
    image_name = item.get("image", "")
    src = _find_image(image_name, raw_dir)
    if not src:
        return

    if module == "bbox":
        dest_dir = Path(output_dir) / "rejected" / "detector"
    elif module in ("plate", "plate_suspect"):
        dest_dir = Path(output_dir) / "rejected" / "ocr"
    elif module == "brand":
        dest_dir = Path(output_dir) / "rejected" / "vehicle_classifier"
    else:
        return

    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dest_dir / src.name)


def run_review(review_dir, raw_dir, output_dir):
    all_items = []
    for module in ("bbox", "brand", "plate"):
        all_items.extend(_load_review_queue(review_dir, module))

    if not all_items:
        print("[review] No items to review.")
        return

    print(f"[review] {len(all_items)} items to review")

    review_log_path = Path(review_dir) / "human" / "review_log.jsonl"
    review_log_path.parent.mkdir(parents=True, exist_ok=True)

    reviewed = 0
    for module, item in all_items:
        image_name = item.get("image", "")
        img_path = _find_image(image_name, raw_dir)

        if img_path:
            img = cv2.imread(str(img_path))
            if img is not None:
                cv2.imshow("Review", img)
                cv2.waitKey(100)

        _display_info(module, item)

        while True:
            action = input("Action: ").strip().lower()
            if action in ("y", "n", "e", "s"):
                break
            print("Invalid. Use y/n/e/s.")

        correction = None
        if action == "e":
            correction = input("Correction: ").strip()
            action = "y"

        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "module": module,
            "image": image_name,
            "action": {"y": "approved", "n": "rejected", "s": "skipped"}[action],
            "correction": correction,
            "original_certainty": item.get("certainty"),
        }
        with open(review_log_path, "a") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        if action == "y":
            _promote(module, item, raw_dir, output_dir, correction)
            print("  -> APPROVED")
        elif action == "n":
            _reject(module, item, raw_dir, output_dir)
            print("  -> REJECTED")
        else:
            print("  -> SKIPPED")

        reviewed += 1

    cv2.destroyAllWindows()
    print(f"[review] Done. Reviewed {reviewed} items.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Human Review Tool")
    p.add_argument("--review-dir", default="./output/review", help="Review directory")
    p.add_argument("--raw-dir", default="./output/raw", help="Raw data directory")
    p.add_argument("--output-dir", default="./output", help="Output directory")
    args = p.parse_args()
    run_review(args.review_dir, args.raw_dir, args.output_dir)
