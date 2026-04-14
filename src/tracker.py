import time
import numpy as np


def _iou(box_a, box_b):
    """Compute IoU between two [x1,y1,x2,y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _centroid(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _box_area(box):
    return (box[2] - box[0]) * (box[3] - box[1])


def _centroid_dist(box_a, box_b):
    ca, cb = _centroid(box_a), _centroid(box_b)
    return ((ca[0] - cb[0]) ** 2 + (ca[1] - cb[1]) ** 2) ** 0.5


def _size_ratio(box_a, box_b):
    a, b = _box_area(box_a), _box_area(box_b)
    if a == 0 or b == 0:
        return 0.0
    return min(a, b) / max(a, b)


def _match_score(det_box, track_box):
    """Combined matching score: higher = better match."""
    iou = _iou(det_box, track_box)
    size_sim = _size_ratio(det_box, track_box)
    diag = ((track_box[2] - track_box[0]) ** 2 + (track_box[3] - track_box[1]) ** 2) ** 0.5
    if diag == 0:
        return 0.0
    dist_norm = _centroid_dist(det_box, track_box) / diag
    dist_score = max(0, 1.0 - dist_norm)
    return 0.4 * iou + 0.3 * size_sim + 0.3 * dist_score


class Tracker:
    """Simple multi-object tracker using centroid + IoU + size matching."""

    MATCH_THRESHOLD = 0.25

    def __init__(self, track_timeout=4.0):
        self.track_timeout = track_timeout
        self.active_tracks = {}  # track_id -> track dict
        self._next_veh_id = 0
        self._next_plt_id = 0

    def _new_id(self, object_type):
        if object_type == "vehicle":
            tid = f"veh_{self._next_veh_id}"
            self._next_veh_id += 1
        else:
            tid = f"plt_{self._next_plt_id}"
            self._next_plt_id += 1
        return tid

    def update(self, detections):
        """Update tracks with new detections. Returns list of active track dicts.

        Each returned dict has: track_id, object_type, bbox_xyxy, conf, class_name,
        first_seen_ts, last_seen_ts, last_saved_ts, save_count, best_quality_score,
        best_crop_path, is_new
        """
        now = time.time()

        # Remove stale tracks
        stale = [tid for tid, t in self.active_tracks.items()
                 if now - t["last_seen_ts"] > self.track_timeout]
        for tid in stale:
            del self.active_tracks[tid]

        # Split detections by type for class-consistent matching
        matched_track_ids = set()
        matched_det_indices = set()

        det_list = list(enumerate(detections))

        # Greedy matching: score all pairs, pick best first
        pairs = []
        for di, det in det_list:
            for tid, trk in self.active_tracks.items():
                if det["object_type"] != trk["object_type"]:
                    continue
                score = _match_score(det["bbox_xyxy"], trk["bbox_xyxy"])
                if score >= self.MATCH_THRESHOLD:
                    pairs.append((score, di, tid))

        pairs.sort(key=lambda x: x[0], reverse=True)

        for score, di, tid in pairs:
            if di in matched_det_indices or tid in matched_track_ids:
                continue
            det = detections[di]
            trk = self.active_tracks[tid]
            trk["bbox_xyxy"] = det["bbox_xyxy"]
            trk["conf"] = det["conf"]
            trk["class_name"] = det["class_name"]
            trk["last_seen_ts"] = now
            trk["is_new"] = False
            matched_track_ids.add(tid)
            matched_det_indices.add(di)

        # Create new tracks for unmatched detections
        for di, det in det_list:
            if di in matched_det_indices:
                continue
            tid = self._new_id(det["object_type"])
            self.active_tracks[tid] = {
                "track_id": tid,
                "object_type": det["object_type"],
                "bbox_xyxy": det["bbox_xyxy"],
                "conf": det["conf"],
                "class_name": det["class_name"],
                "first_seen_ts": now,
                "last_seen_ts": now,
                "last_saved_ts": None,
                "save_count": 0,
                "best_quality_score": 0.0,
                "best_crop_path": None,
                "is_new": True,
            }

        return list(self.active_tracks.values())
