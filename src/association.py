def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _overlap_ratio(plate_box, vehicle_box):
    x1 = max(plate_box[0], vehicle_box[0])
    y1 = max(plate_box[1], vehicle_box[1])
    x2 = min(plate_box[2], vehicle_box[2])
    y2 = min(plate_box[3], vehicle_box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    plate_area = (plate_box[2] - plate_box[0]) * (plate_box[3] - plate_box[1])
    return inter / plate_area if plate_area > 0 else 0.0


def _x_overlap_ratio(plate_box, vehicle_box):
    overlap = min(plate_box[2], vehicle_box[2]) - max(plate_box[0], vehicle_box[0])
    plate_w = plate_box[2] - plate_box[0]
    return max(0, overlap) / plate_w if plate_w > 0 else 0.0


def _score_match(plate_box, vehicle_box):
    pcx, pcy = _center(plate_box)
    vx1, vy1, vx2, vy2 = vehicle_box
    vw = vx2 - vx1
    vh = vy2 - vy1
    inside = vx1 <= pcx <= vx2 and vy1 <= pcy <= vy2
    # If plate center is not inside the vehicle, no score — use proximity branch instead
    if not inside:
        return 0.0
    containment = 1.0
    overlap = _overlap_ratio(plate_box, vehicle_box)
    if vh > 0:
        y_frac = (pcy - vy1) / vh
        lower_bonus = y_frac
    else:
        lower_bonus = 0.0
    plate_area = (plate_box[2] - plate_box[0]) * (plate_box[3] - plate_box[1])
    vehicle_area = vw * vh
    if vehicle_area > 0:
        size_ratio = plate_area / vehicle_area
        size_ok = 1.0 if size_ratio < 0.15 else max(0, 1.0 - (size_ratio - 0.15) * 5)
    else:
        size_ok = 0.0
    return 0.3 * containment + 0.25 * overlap + 0.25 * lower_bonus + 0.2 * size_ok


def associate_plates(vehicles, plates, proximity_margin_px=60):
    result = {}
    for plate in plates:
        pbox = plate["bbox_xyxy"]
        pid = plate["track_id"]
        candidates = []
        for veh in vehicles:
            vbox = veh["bbox_xyxy"]
            score = _score_match(pbox, vbox)
            if score > 0.15:
                candidates.append((score, veh["track_id"], "matched"))
        if not candidates:
            ptop = pbox[1]
            for veh in vehicles:
                vbox = veh["bbox_xyxy"]
                vbottom = vbox[3]
                gap = ptop - vbottom
                if 0 <= gap <= proximity_margin_px:
                    x_overlap = _x_overlap_ratio(pbox, vbox)
                    if x_overlap >= 0.5:
                        candidates.append((0.1, veh["track_id"], "matched_provisional"))
        if not candidates:
            result[pid] = {"status": "unmatched", "vehicle_track_id": None, "score": 0.0}
        elif len(candidates) == 1:
            score, vid, status = candidates[0]
            result[pid] = {"status": status, "vehicle_track_id": vid, "score": score}
        else:
            candidates.sort(key=lambda x: x[0], reverse=True)
            best_score = candidates[0][0]
            second_score = candidates[1][0]
            if best_score > second_score * 1.5:
                score, vid, status = candidates[0]
                result[pid] = {"status": status, "vehicle_track_id": vid, "score": score}
            else:
                result[pid] = {"status": "ambiguous", "vehicle_track_id": None, "score": best_score}
    return result
