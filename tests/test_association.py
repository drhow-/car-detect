from src.association import associate_plates


def _trk(track_id, x1, y1, x2, y2, obj_type):
    return {
        "track_id": track_id,
        "object_type": obj_type,
        "bbox_xyxy": [x1, y1, x2, y2],
        "conf": 0.8,
        "class_name": "car" if obj_type == "vehicle" else "license_plate",
    }


def test_plate_inside_vehicle_matched():
    vehicles = [_trk("veh_0", 100, 100, 500, 400, "vehicle")]
    plates = [_trk("plt_0", 200, 350, 350, 390, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "matched"
    assert result["plt_0"]["vehicle_track_id"] == "veh_0"


def test_plate_no_vehicle_unmatched():
    vehicles = []
    plates = [_trk("plt_0", 200, 350, 350, 390, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "unmatched"


def test_plate_ambiguous_multiple_vehicles():
    vehicles = [
        _trk("veh_0", 100, 100, 350, 400, "vehicle"),
        _trk("veh_1", 300, 100, 550, 400, "vehicle"),
    ]
    plates = [_trk("plt_0", 310, 350, 340, 390, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] in ("matched", "ambiguous")


def test_plate_below_vehicle_provisional():
    vehicles = [_trk("veh_0", 100, 100, 500, 300, "vehicle")]
    plates = [_trk("plt_0", 200, 310, 350, 340, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "matched_provisional"
    assert result["plt_0"]["vehicle_track_id"] == "veh_0"


def test_plate_too_far_below_unmatched():
    vehicles = [_trk("veh_0", 100, 100, 500, 300, "vehicle")]
    plates = [_trk("plt_0", 200, 400, 350, 430, "plate")]
    result = associate_plates(vehicles, plates, proximity_margin_px=60)
    assert result["plt_0"]["status"] == "unmatched"
