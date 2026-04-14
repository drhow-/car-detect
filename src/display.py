# src/display.py
import cv2


class Display:
    """OpenCV live preview with bounding boxes, association lines, and counters."""

    VEHICLE_COLOR = (0, 255, 0)   # Green
    PLATE_COLOR = (0, 0, 255)     # Red
    LINE_COLOR = (255, 200, 0)    # Cyan-ish
    FONT = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(self):
        self.window_name = "Syrian Plate Collector"

    def draw(self, frame, tracks, associations, stats):
        """Draw boxes, labels, association lines, and stats overlay.

        Args:
            frame: BGR numpy array (will be modified in-place)
            tracks: list of track dicts with bbox_xyxy, object_type, track_id
            associations: dict from associate_plates() — plate_track_id -> {status, vehicle_track_id}
            stats: dict with keys: vehicles, plates, active_tracks, saved, rejected
        Returns:
            frame with drawings
        """
        vis = frame.copy()
        track_by_id = {t["track_id"]: t for t in tracks}

        for trk in tracks:
            x1, y1, x2, y2 = [int(v) for v in trk["bbox_xyxy"]]
            if trk["object_type"] == "vehicle":
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.VEHICLE_COLOR, 2)
                label = f"VEHICLE {trk['track_id']}"
                cv2.putText(vis, label, (x1, y1 - 5), self.FONT, 0.5, self.VEHICLE_COLOR, 1)
            else:
                cv2.rectangle(vis, (x1, y1), (x2, y2), self.PLATE_COLOR, 2)
                label = f"PLATE {trk['track_id']}"
                cv2.putText(vis, label, (x1, y1 - 5), self.FONT, 0.5, self.PLATE_COLOR, 1)

        for pid, assoc in associations.items():
            if assoc["vehicle_track_id"] and pid in track_by_id and assoc["vehicle_track_id"] in track_by_id:
                pbox = track_by_id[pid]["bbox_xyxy"]
                vbox = track_by_id[assoc["vehicle_track_id"]]["bbox_xyxy"]
                pcx = int((pbox[0] + pbox[2]) / 2)
                pcy = int((pbox[1] + pbox[3]) / 2)
                vcx = int((vbox[0] + vbox[2]) / 2)
                vcy = int((vbox[1] + vbox[3]) / 2)
                cv2.line(vis, (pcx, pcy), (vcx, vcy), self.LINE_COLOR, 1)

        y = 25
        for key, val in stats.items():
            text = f"{key}: {val}"
            cv2.putText(vis, text, (10, y), self.FONT, 0.6, (255, 255, 255), 2)
            cv2.putText(vis, text, (10, y), self.FONT, 0.6, (0, 0, 0), 1)
            y += 25

        return vis

    def show(self, frame):
        """Display frame and return True if user pressed 'q'."""
        cv2.imshow(self.window_name, frame)
        key = cv2.waitKey(1) & 0xFF
        return key == ord("q")

    def close(self):
        cv2.destroyAllWindows()
