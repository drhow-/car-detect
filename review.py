# review.py
"""AI Review CLI — run bbox, brand, or plate review on collected data."""
import argparse
import json
import sys
import time
from pathlib import Path

import cv2


def _collect_images(image_arg, dir_arg, extensions=("*.jpg", "*.png")):
    paths = []
    if image_arg:
        from glob import glob
        paths = [Path(p) for p in glob(image_arg)]
    elif dir_arg:
        d = Path(dir_arg)
        for ext in extensions:
            paths.extend(sorted(d.rglob(ext)))
    return paths


def _get_client():
    import anthropic
    return anthropic.Anthropic()


def _append_results(results, output_file):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "a") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _batch(items, size):
    for i in range(0, len(items), size):
        yield items[i:i + size]


def cmd_bbox(args):
    from src.review.bbox import verify_bbox_batch
    from src.review.grid import parse_grid_size

    paths = _collect_images(args.image, args.dir, ("*.jpg",))
    if not paths:
        print("[review] No images found.")
        return

    client = _get_client()
    rows, cols = parse_grid_size(args.bbox_grid_size)
    batch_size = rows * cols
    output_file = Path(args.output_dir) / "review" / "ai" / "bbox_results.jsonl"

    print(f"[review:bbox] Processing {len(paths)} frames in batches of {batch_size}")
    for i, batch_paths in enumerate(_batch(paths, batch_size)):
        print(f"  Batch {i + 1}...")
        label_dir = batch_paths[0].parent.parent.parent / "labels" / batch_paths[0].parent.name
        results = verify_bbox_batch(batch_paths, label_dir, client,
                                    grid_size=args.bbox_grid_size, tile_px=args.tile_px)
        _append_results(results, output_file)
        time.sleep(1)

    print(f"[review:bbox] Results saved to {output_file}")


def cmd_brand(args):
    from src.review.brand import detect_brand_batch
    from src.review.grid import parse_grid_size

    paths = _collect_images(args.image, args.dir, ("*.png",))
    if not paths:
        print("[review] No images found.")
        return

    client = _get_client()
    rows, cols = parse_grid_size(args.grid_size)
    batch_size = rows * cols
    output_file = Path(args.output_dir) / "review" / "ai" / "brand_results.jsonl"

    print(f"[review:brand] Processing {len(paths)} crops in batches of {batch_size}")
    for i, batch_paths in enumerate(_batch(paths, batch_size)):
        print(f"  Batch {i + 1}...")
        results = detect_brand_batch(batch_paths, client,
                                     grid_size=args.grid_size, tile_px=args.tile_px)
        _append_results(results, output_file)
        time.sleep(1)

    print(f"[review:brand] Results saved to {output_file}")


def cmd_plate(args):
    from src.review.plate import read_plate_batch
    from src.review.grid import parse_grid_size

    paths = _collect_images(args.image, args.dir, ("*.png",))
    if not paths:
        print("[review] No images found.")
        return

    client = _get_client()
    rows, cols = parse_grid_size(args.grid_size)
    batch_size = rows * cols
    output_file = Path(args.output_dir) / "review" / "ai" / "plate_results.jsonl"
    suspect_file = Path(args.output_dir) / "review" / "queues" / "plate_format_suspect.jsonl"

    print(f"[review:plate] Processing {len(paths)} crops in batches of {batch_size}")
    for i, batch_paths in enumerate(_batch(paths, batch_size)):
        print(f"  Batch {i + 1}...")
        results = read_plate_batch(batch_paths, client,
                                   grid_size=args.grid_size, tile_px=args.tile_px)

        valid = [r for r in results if r.get("format_valid", True)]
        suspect = [r for r in results if not r.get("format_valid", True)]

        _append_results(valid, output_file)
        if suspect:
            _append_results(suspect, suspect_file)
            print(f"    {len(suspect)} format-suspect items queued for review")

        time.sleep(1)

    print(f"[review:plate] Results saved to {output_file}")


def cmd_all(args):
    print("[review] Running all three modules...")
    base = Path(args.output_dir)

    args.dir = str(base / "raw" / "frames")
    args.image = None
    cmd_bbox(args)

    args.dir = str(base / "raw" / "crops" / "vehicles")
    cmd_brand(args)

    args.dir = str(base / "raw" / "crops" / "plates")
    cmd_plate(args)


def main():
    p = argparse.ArgumentParser(description="AI Review Pipeline")
    sub = p.add_subparsers(dest="command", required=True)

    for name, func in [("bbox", cmd_bbox), ("brand", cmd_brand),
                        ("plate", cmd_plate), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        sp.add_argument("--image", default=None, help="Glob pattern for images")
        sp.add_argument("--dir", default=None, help="Directory of images")
        sp.add_argument("--output-dir", default="./output", help="Output directory")
        sp.add_argument("--grid-size", default="3x3", help="Crops per composite")
        sp.add_argument("--bbox-grid-size", default="2x2", help="Frames per composite (bbox only)")
        sp.add_argument("--tile-px", type=int, default=512, help="Per-cell tile size")
        sp.set_defaults(func=func)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
