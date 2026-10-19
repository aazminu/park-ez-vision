"""Unit and integration test suite for Park-EZ (1st Place SMU OIT Challenge @ HackSMU VI)."""

from __future__ import annotations

import pytest

from park_ez.occupancy import (
    BoundingBox,
    ParkingStall,
    TemporalOccupancyTracker,
    compute_bbox_iou,
    compute_polygon_iou,
    polygon_area,
)
from park_ez.router import (
    CampusNode,
    ParkingLotCandidate,
    RoadEdge,
    estimate_stall_capture_probability,
    recommend_parking_lots,
    shortest_driving_path_astar,
)


def _build_smu_campus_graph() -> tuple[dict[str, CampusNode], dict[str, list[RoadEdge]]]:
    nodes = {
        "GATE_MOCKINGBIRD": CampusNode("GATE_MOCKINGBIRD", (0.0, 0.0)),
        "BISHOP_BLVD": CampusNode("BISHOP_BLVD", (200.0, 0.0)),
        "LOT_BINKLEY": CampusNode("LOT_BINKLEY", (200.0, 150.0)),
        "LOT_MOODY": CampusNode("LOT_MOODY", (450.0, 50.0)),
        "LOT_MEADOWS": CampusNode("LOT_MEADOWS", (150.0, -120.0)),
    }
    adjacency: dict[str, list[RoadEdge]] = {
        "GATE_MOCKINGBIRD": [
            RoadEdge("BISHOP_BLVD", distance_m=200.0, speed_limit_mps=10.0),
            RoadEdge("LOT_MEADOWS", distance_m=192.0, speed_limit_mps=8.0),
        ],
        "BISHOP_BLVD": [
            RoadEdge("LOT_BINKLEY", distance_m=150.0, speed_limit_mps=10.0),
            RoadEdge("LOT_MOODY", distance_m=260.0, speed_limit_mps=10.0),
        ],
        "LOT_BINKLEY": [],
        "LOT_MOODY": [],
        "LOT_MEADOWS": [],
    }
    return nodes, adjacency


def test_compute_bbox_iou_identical_disjoint_and_partial() -> None:
    box_a = BoundingBox(0.0, 0.0, 10.0, 10.0)
    box_b = BoundingBox(0.0, 0.0, 10.0, 10.0)
    box_disjoint = BoundingBox(20.0, 20.0, 30.0, 30.0)
    box_half = BoundingBox(5.0, 0.0, 15.0, 10.0)

    assert compute_bbox_iou(box_a, box_b) == pytest.approx(1.0)
    assert compute_bbox_iou(box_a, box_disjoint) == pytest.approx(0.0)
    # Intersection = 5x10 = 50, Union = 100 + 100 - 50 = 150 -> IoU = 1/3
    assert compute_bbox_iou(box_a, box_half) == pytest.approx(1.0 / 3.0)


def test_bounding_box_validation() -> None:
    with pytest.raises(ValueError, match="max coordinates"):
        BoundingBox(10.0, 0.0, 5.0, 10.0)
    with pytest.raises(ValueError, match="confidence"):
        BoundingBox(0.0, 0.0, 5.0, 5.0, confidence=1.5)


def test_polygon_area_and_oriented_polygon_iou() -> None:
    poly_a = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    poly_b = [(5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0)]
    assert polygon_area(poly_a) == pytest.approx(100.0)
    assert compute_polygon_iou(poly_a, poly_b) == pytest.approx(1.0 / 3.0)
    assert compute_polygon_iou(poly_a, poly_a) == pytest.approx(1.0)


def test_temporal_hysteresis_rejects_single_frame_occlusion_flicker() -> None:
    stall = ParkingStall(
        stall_id="S-01",
        bbox=BoundingBox(0.0, 0.0, 10.0, 20.0),
    )
    tracker = TemporalOccupancyTracker(
        iou_threshold=0.40,
        enter_occupied_frames=3,
        exit_vacant_frames=3,
    )
    car_det = BoundingBox(0.5, 0.5, 9.5, 19.5, confidence=0.95)

    # Frame 1 & 2 with vehicle present -> still debouncing (not yet occupied)
    s1 = tracker.update_frame([stall], [car_det])["S-01"]
    assert not s1.is_occupied
    s2 = tracker.update_frame([stall], [car_det])["S-01"]
    assert not s2.is_occupied

    # Frame 3 -> transitions to occupied
    s3 = tracker.update_frame([stall], [car_det])["S-01"]
    assert s3.is_occupied
    assert s3.kalman_occupancy_prob > 0.70

    # Frame 4: single-frame pedestrian occlusion drops detection -> remains occupied!
    s4 = tracker.update_frame([stall], [])["S-01"]
    assert s4.is_occupied
    assert s4.consecutive_vacant_frames == 1

    # Frame 5 & 6: two more vacant frames -> transitions back to vacant on 3rd empty frame
    tracker.update_frame([stall], [])
    s6 = tracker.update_frame([stall], [])["S-01"]
    assert not s6.is_occupied


def test_ada_unauthorized_vehicle_violation_detection() -> None:
    ada_stall = ParkingStall(
        stall_id="ADA-01",
        bbox=BoundingBox(0.0, 0.0, 10.0, 20.0),
        is_ada_accessible=True,
        is_van_accessible=True,
    )
    tracker = TemporalOccupancyTracker(enter_occupied_frames=2)

    # Vehicle without ADA placard parks for 2 frames
    unauth_car = BoundingBox(0.0, 0.0, 10.0, 20.0, has_ada_placard_tag=False)
    tracker.update_frame([ada_stall], [unauth_car])
    state = tracker.update_frame([ada_stall], [unauth_car])["ADA-01"]

    assert state.is_occupied
    assert state.ada_violation
    assert state.violation_reason == "UNAUTHORIZED_NO_ADA_PERMIT"


def test_ada_authorized_placard_vehicle_has_no_violation() -> None:
    ada_stall = ParkingStall(
        stall_id="ADA-02",
        bbox=BoundingBox(0.0, 0.0, 10.0, 20.0),
        is_ada_accessible=True,
    )
    tracker = TemporalOccupancyTracker(enter_occupied_frames=1)
    auth_car = BoundingBox(0.0, 0.0, 10.0, 20.0, has_ada_placard_tag=True)
    state = tracker.update_frame([ada_stall], [auth_car])["ADA-02"]

    assert state.is_occupied
    assert not state.ada_violation
    assert state.violation_reason is None


def test_ada_van_access_aisle_encroachment_violation() -> None:
    van_stall = ParkingStall(
        stall_id="VAN-01",
        bbox=BoundingBox(0.0, 0.0, 10.0, 20.0),
        is_ada_accessible=True,
        is_van_accessible=True,
        access_aisle_bbox=BoundingBox(10.0, 0.0, 15.0, 20.0),
    )
    tracker = TemporalOccupancyTracker(
        enter_occupied_frames=2, aisle_encroachment_iou_threshold=0.08
    )

    # Vehicle encroaching into the striped wheelchair access aisle
    encroaching_car = BoundingBox(11.0, 2.0, 18.0, 18.0, has_ada_placard_tag=True)
    state = tracker.update_frame([van_stall], [encroaching_car])["VAN-01"]

    assert state.ada_violation
    assert state.violation_reason == "ACCESS_AISLE_ENCROACHMENT"


def test_lot_occupancy_summary_counts() -> None:
    stalls = [
        ParkingStall("REG-1", BoundingBox(0.0, 0.0, 10.0, 20.0)),
        ParkingStall("REG-2", BoundingBox(20.0, 0.0, 30.0, 20.0)),
        ParkingStall(
            "ADA-VAN-1",
            BoundingBox(40.0, 0.0, 50.0, 20.0),
            is_ada_accessible=True,
            is_van_accessible=True,
        ),
    ]
    tracker = TemporalOccupancyTracker(enter_occupied_frames=1)
    # Occupy REG-1 only
    tracker.update_frame(stalls, [BoundingBox(0.0, 0.0, 10.0, 20.0)])
    summary = tracker.summarize_lot(stalls)

    assert summary.total_stalls == 3
    assert summary.occupied_stalls == 1
    assert summary.vacant_regular_stalls == 1
    assert summary.vacant_ada_stalls == 1
    assert summary.vacant_van_stalls == 1
    assert summary.active_ada_violations == 0


def test_oriented_polygon_stall_matching_in_tracker() -> None:
    angled_poly = ((0.0, 0.0), (10.0, 2.0), (10.0, 12.0), (0.0, 10.0))
    stall = ParkingStall(
        stall_id="ANGLED-1",
        bbox=BoundingBox(0.0, 0.0, 10.0, 12.0),
        polygon=angled_poly,
    )
    tracker = TemporalOccupancyTracker(enter_occupied_frames=1, iou_threshold=0.35)
    det = BoundingBox(1.0, 2.0, 9.0, 10.0)
    state = tracker.update_frame([stall], [det])["ANGLED-1"]
    assert state.is_occupied
    assert state.raw_max_iou > 0.40


def test_shortest_driving_path_astar_finds_optimal_route() -> None:
    nodes, adj = _build_smu_campus_graph()
    path, dist_m, eta_s = shortest_driving_path_astar(
        nodes, adj, "GATE_MOCKINGBIRD", "LOT_BINKLEY"
    )
    assert path == ["GATE_MOCKINGBIRD", "BISHOP_BLVD", "LOT_BINKLEY"]
    assert dist_m == pytest.approx(350.0)
    assert eta_s == pytest.approx(35.0)


def test_shortest_driving_path_astar_unreachable_raises() -> None:
    nodes, adj = _build_smu_campus_graph()
    with pytest.raises(ValueError, match="No directed driving path"):
        shortest_driving_path_astar(nodes, adj, "LOT_BINKLEY", "GATE_MOCKINGBIRD")


def test_estimate_stall_capture_probability_monotonicity() -> None:
    p_high = estimate_stall_capture_probability(
        vacant_stalls=12,
        total_stalls=50,
        hourly_turnover_rate=20.0,
        driving_eta_seconds=60.0,
    )
    p_low = estimate_stall_capture_probability(
        vacant_stalls=1,
        total_stalls=50,
        hourly_turnover_rate=2.0,
        driving_eta_seconds=600.0,
    )
    assert p_high > p_low
    assert 0.0 < p_low < p_high <= 0.999


def test_recommend_parking_lots_standard_vs_ada_van_constraints() -> None:
    nodes, adj = _build_smu_campus_graph()
    dallas_hall_xy = (220.0, 160.0)

    lots = [
        # Lot Binkley: close to Dallas Hall, steep hill (6.8% grade -> exceeds 5% ADA limit)
        ParkingLotCandidate(
            lot_id="LOT_BINKLEY",
            node_id="LOT_BINKLEY",
            xy=(200.0, 150.0),
            total_stalls=120,
            vacant_regular_stalls=18,
            vacant_ada_stalls=2,
            vacant_van_stalls=0,
            hourly_turnover_rate=25.0,
            walk_slope_pct_to_dest=6.8,
            has_curb_cut_ramp=True,
        ),
        # Lot Moody: gentle 1.8% grade, curb-cut ramp, van-accessible stalls open
        ParkingLotCandidate(
            lot_id="LOT_MOODY",
            node_id="LOT_MOODY",
            xy=(450.0, 50.0),
            total_stalls=200,
            vacant_regular_stalls=30,
            vacant_ada_stalls=4,
            vacant_van_stalls=2,
            hourly_turnover_rate=40.0,
            walk_slope_pct_to_dest=1.8,
            has_curb_cut_ramp=True,
        ),
        # Lot Meadows: lacks curb-cut ramp
        ParkingLotCandidate(
            lot_id="LOT_MEADOWS",
            node_id="LOT_MEADOWS",
            xy=(150.0, -120.0),
            total_stalls=80,
            vacant_regular_stalls=10,
            vacant_ada_stalls=2,
            vacant_van_stalls=1,
            hourly_turnover_rate=15.0,
            walk_slope_pct_to_dest=2.0,
            has_curb_cut_ramp=False,
        ),
    ]

    # Standard commuter: Lot Binkley is closest to Dallas Hall and wins
    reg_recs = recommend_parking_lots(
        nodes,
        adj,
        lots,
        origin_node="GATE_MOCKINGBIRD",
        target_building_xy=dallas_hall_xy,
        requires_ada=False,
    )
    assert len(reg_recs) == 3
    assert reg_recs[0].lot_id == "LOT_BINKLEY"

    # ADA Van-Accessible commuter: Lot Binkley (6.8% > 5% slope, 0 van stalls) and
    # Lot Meadows (no curb cut) are filtered out; Lot Moody is recommended!
    van_recs = recommend_parking_lots(
        nodes,
        adj,
        lots,
        origin_node="GATE_MOCKINGBIRD",
        target_building_xy=dallas_hall_xy,
        requires_ada=True,
        requires_van_accessible=True,
        max_ada_walk_slope_pct=5.0,
    )
    assert len(van_recs) == 1
    assert van_recs[0].lot_id == "LOT_MOODY"


def test_tracker_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="iou_threshold"):
        TemporalOccupancyTracker(iou_threshold=1.2)
    with pytest.raises(ValueError, match="Hysteresis"):
        TemporalOccupancyTracker(enter_occupied_frames=0)
