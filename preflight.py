# preflight.py
"""Pre-flight validation: test plate detector recall on sample images."""
import argparse
from pathlib import Path

from ultralytics import YOLO


def run_preflight(plate_model_path, test_dir, conf=0.3, imgsz=960):
    model = YOLO(plate_model_path)
    test_dir = Path(test_dir)
    images = sorted(test_dir.glob("*.jpg")) + sorted(test_dir.glob("*.png"))

    if not images:
        print(f"[preflight] No images found in {test_dir}")
        return

    total = len(images)
    detected = 0
    all_confs = []

    print(f"[preflight] Testing {total} images with model: {plate_model_path}")
    print(f"[preflight] Confidence threshold: {conf}")
    print("-" * 60)

    for img_path in images:
        results = model(str(img_path), conf=conf, imgsz=imgsz, verbose=False)
        n_plates = 0
        for r in results:
            n_plates += len(r.boxes)
            for box in r.boxes:
                all_confs.append(float(box.conf[0]))

        status = "OK" if n_plates > 0 else "MISS"
        if n_plates > 0:
            detected += 1
        print(f"  [{status}] {img_path.name}: {n_plates} plate(s)")

    recall = detected / total * 100
    print("-" * 60)
    print(f"[preflight] Recall: {detected}/{total} = {recall:.0f}%")
    if all_confs:
        avg_conf = sum(all_confs) / len(all_confs)
        print(f"[preflight] Avg confidence: {avg_conf:.2f}")

    if recall >= 70:
        print("[preflight] PASS — proceed with this model.")
    elif recall >= 50:
        print("[preflight] WARN — proceed but flag; consider fine-tuning sooner.")
    else:
        print("[preflight] FAIL — swap to fallback model before collection.")

    return recall


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Pre-flight plate detector validation")
    p.add_argument("--plate-model", required=True, help="Plate model path or HF repo")
    p.add_argument("--test-dir", required=True, help="Directory of test plate images")
    p.add_argument("--conf", type=float, default=0.3)
    p.add_argument("--imgsz", type=int, default=960)
    args = p.parse_args()
    run_preflight(args.plate_model, args.test_dir, args.conf, args.imgsz)
