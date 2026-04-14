from ultralytics import YOLO

# COCO class IDs for vehicles
_VEHICLE_COCO_IDS = {2: "car", 5: "bus", 7: "truck"}


class Detector:
    """Runs two YOLO models (vehicle + plate) on a frame."""

    def __init__(self, vehicle_model_path, plate_model_path,
                 vehicle_conf=0.3, plate_conf=0.3, imgsz=960):
        self.vehicle_model = YOLO(vehicle_model_path)
        self.plate_model = YOLO(plate_model_path)
        self.vehicle_conf = vehicle_conf
        self.plate_conf = plate_conf
        self.imgsz = imgsz

    def detect(self, frame):
        """Run both detectors on frame. Returns list of detection dicts.

        Each dict has: bbox_xyxy, conf, class_name, object_type
        bbox_xyxy is in pixel coordinates of the input frame.
        """
        detections = []

        # Vehicle detection
        veh_results = self.vehicle_model(
            frame, conf=self.vehicle_conf, imgsz=self.imgsz, verbose=False
        )
        for r in veh_results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                if cls_id in _VEHICLE_COCO_IDS:
                    detections.append({
                        "bbox_xyxy": box.xyxy[0].cpu().numpy().tolist(),
                        "conf": float(box.conf[0]),
                        "class_name": _VEHICLE_COCO_IDS[cls_id],
                        "object_type": "vehicle",
                    })

        # Plate detection
        plt_results = self.plate_model(
            frame, conf=self.plate_conf, imgsz=self.imgsz, verbose=False
        )
        for r in plt_results:
            for box in r.boxes:
                detections.append({
                    "bbox_xyxy": box.xyxy[0].cpu().numpy().tolist(),
                    "conf": float(box.conf[0]),
                    "class_name": "license_plate",
                    "object_type": "plate",
                })

        return detections
