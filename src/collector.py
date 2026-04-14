# src/collector.py
import time
import uuid
from datetime import datetime, timezone

import cv2

from src.config import parse_args
from src.capture import FrameGrabber
from src.detector import Detector
from src.tracker import Tracker
from src.association import associate_plates
from src.quality import check_quality
from src.saver import SaveDecider, save_frame, save_crop, save_label, extract_crop
from src.metadata import MetadataLogger
from src.display import Display


def _generate_session_id(camera_id):
    date_str = datetime.now().strftime("%Y%m%d")
    short_uuid = uuid.uuid4().hex[:4]
    return f"sess_{date_str}_{camera_id}_{short_uuid}"


def run(argv=None):
    args = parse_args(argv)

    camera_id = f"cam{args.camera}"
    session_id = args.session_id or _generate_session_id(camera_id)
    print(f"[collector] Session: {session_id}")
    print(f"[collector] Output: {args.output}")

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera {args.camera}")
        return

    grabber = FrameGrabber(cap, frame_stride=args.frame_stride)
    print(f"[collector] Camera {args.camera}: {grabber.width}x{grabber.height} @ {grabber.fps:.0f}fps")
    print(f"[collector] Frame stride: {args.frame_stride} (~{grabber.fps / args.frame_stride:.0f} inference FPS)")

    detector = Detector(
        args.vehicle_model, args.plate_model,
        vehicle_conf=args.vehicle_conf, plate_conf=args.plate_conf,
        imgsz=args.imgsz,
    )
    tracker = Tracker(track_timeout=args.track_timeout)
    save_decider = SaveDecider(
        vehicle_cooldown=args.vehicle_cooldown,
        plate_cooldown=args.plate_cooldown,
        save_better_only=args.save_better_only,
    )
    meta = MetadataLogger(args.output)
    meta.ensure_classes_txt()
    display = Display() if not args.no_display else None

    stats = {
        "vehicles": 0, "plates": 0, "active_tracks": 0,
        "saved": 0, "rejected": 0,
    }
    session_start = datetime.now(timezone.utc)
    total_frames = 0
    last_tracks = []
    last_associations = {}

    print("[collector] Running... Press 'q' to quit.")

    try:
        while True:
            frame, should_process, frame_idx = grabber.next()
            if frame is None:
                break
            total_frames += 1

            if should_process:
                t0 = time.time()

                detections = detector.detect(frame)
                tracks = tracker.update(detections)
                last_tracks = tracks

                vehicle_tracks = [t for t in tracks if t["object_type"] == "vehicle"]
                plate_tracks = [t for t in tracks if t["object_type"] == "plate"]

                associations = associate_plates(
                    vehicle_tracks, plate_tracks,
                    proximity_margin_px=args.proximity_margin_px,
                )
                last_associations = associations

                stats["vehicles"] = len(vehicle_tracks)
                stats["plates"] = len(plate_tracks)
                stats["active_tracks"] = len(tracks)

                timestamp_ms = int(time.time() * 1000)
                save_worthy_dets = []

                for trk in tracks:
                    bbox = trk["bbox_xyxy"]
                    padding = 0.10 if trk["object_type"] == "vehicle" else 0.05
                    crop = extract_crop(frame, bbox, padding_frac=padding)

                    if crop.size == 0:
                        continue

                    qr = check_quality(
                        crop, trk["object_type"], bbox, frame.shape,
                        min_plate_w=args.min_plate_w,
                        min_plate_h=args.min_plate_h,
                        min_vehicle_area=args.min_vehicle_area,
                        min_blur_score=args.min_blur_score,
                    )

                    if not qr["passes"]:
                        stats["rejected"] += 1
                        continue

                    if not save_decider.should_save(trk, qr["quality_score"]):
                        continue

                    crop_path = save_crop(
                        crop, args.output, session_id,
                        trk["track_id"], timestamp_ms, trk["object_type"],
                    )

                    trk["last_saved_ts"] = time.time()
                    trk["save_count"] += 1
                    if qr["quality_score"] > trk["best_quality_score"]:
                        trk["best_quality_score"] = qr["quality_score"]
                        trk["best_crop_path"] = str(crop_path)

                    det_record = {
                        "object_type": trk["object_type"],
                        "track_id": trk["track_id"],
                        "vehicle_type": trk["class_name"] if trk["object_type"] == "vehicle" else None,
                        "bbox_xyxy": [int(v) for v in bbox],
                        "detector_conf": trk["conf"],
                        "crop_path": str(crop_path),
                        "quality_score": round(qr["quality_score"], 3),
                        "blur_score": round(qr["blur_score"], 1),
                        "truncated": qr["truncated"],
                        "occluded": False,
                        "review_status": "pending",
                        "negative_type": None,
                    }

                    if trk["object_type"] == "plate" and trk["track_id"] in associations:
                        assoc = associations[trk["track_id"]]
                        det_record["association_status"] = assoc["status"]
                        det_record["associated_vehicle_track_id"] = assoc["vehicle_track_id"]
                        det_record["association_score"] = round(assoc["score"], 3)
                    elif trk["object_type"] == "vehicle":
                        linked_plate = None
                        for pid, a in associations.items():
                            if a["vehicle_track_id"] == trk["track_id"]:
                                linked_plate = pid
                                break
                        det_record["associated_plate_track_id"] = linked_plate

                    save_worthy_dets.append(det_record)

                if save_worthy_dets:
                    frame_path = save_frame(frame, args.output, session_id, timestamp_ms, frame_idx)
                    label_path = save_label(
                        save_worthy_dets, frame.shape[1], frame.shape[0],
                        args.output, session_id, timestamp_ms, frame_idx,
                    )

                    event = {
                        "event_id": meta.next_event_id(),
                        "session_id": session_id,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "camera_id": camera_id,
                        "frame_path": str(frame_path),
                        "label_path": str(label_path),
                        "image_width": frame.shape[1],
                        "image_height": frame.shape[0],
                        "detections": save_worthy_dets,
                    }
                    meta.log_save_event(event)
                    stats["saved"] += len(save_worthy_dets)

                last_inference_time = time.time() - t0
                frame_budget = 1.0 / (grabber.fps / grabber.frame_stride)
                if args.adaptive_stride and last_inference_time > frame_budget * 1.5:
                    new_stride = grabber.raise_stride()
                    print(f"[collector] Adaptive stride raised to {new_stride}")

            if display:
                vis = display.draw(frame, last_tracks, last_associations, stats)
                if display.show(vis):
                    break

    except KeyboardInterrupt:
        print("\n[collector] Interrupted.")
    finally:
        session_end = datetime.now(timezone.utc)
        meta.log_session({
            "session_id": session_id,
            "camera_id": camera_id,
            "start_time": session_start.isoformat(),
            "end_time": session_end.isoformat(),
            "duration_seconds": (session_end - session_start).total_seconds(),
            "total_frames": total_frames,
            "total_saved": stats["saved"],
            "total_rejected": stats["rejected"],
        })

        for trk in tracker.active_tracks.values():
            meta.log_track({
                "track_id": trk["track_id"],
                "object_type": trk["object_type"],
                "first_seen_ts": trk["first_seen_ts"],
                "last_seen_ts": trk["last_seen_ts"],
                "save_count": trk["save_count"],
                "best_quality_score": trk["best_quality_score"],
                "best_crop_path": trk["best_crop_path"],
            })

        grabber.release()
        if display:
            display.close()
        print(f"[collector] Done. {stats['saved']} saves, {stats['rejected']} rejected, {total_frames} frames.")


if __name__ == "__main__":
    run()
