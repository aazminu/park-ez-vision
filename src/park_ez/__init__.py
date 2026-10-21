"""Park-EZ: Real-Time Campus Parking Occupancy Geometry & Accessible Lot Routing.

1st Place SMU OIT Challenge @ HackSMU VI.
Author: Abdullah Azmi (https://abdullahazmi.com)
"""

from park_ez.occupancy import (
    BoundingBox,
    LotOccupancySummary,
    ParkingStall,
    StallTrackState,
    TemporalOccupancyTracker,
    compute_bbox_iou,
    compute_polygon_iou,
    polygon_area,
)
from park_ez.router import (
    CampusNode,
    LotRecommendation,
    ParkingLotCandidate,
    RoadEdge,
    estimate_stall_capture_probability,
    recommend_parking_lots,
    shortest_driving_path_astar,
)

__all__ = [
    "BoundingBox",
    "CampusNode",
    "LotOccupancySummary",
    "LotRecommendation",
    "ParkingLotCandidate",
    "ParkingStall",
    "RoadEdge",
    "StallTrackState",
    "TemporalOccupancyTracker",
    "compute_bbox_iou",
    "compute_polygon_iou",
    "estimate_stall_capture_probability",
    "polygon_area",
    "recommend_parking_lots",
    "shortest_driving_path_astar",
]

__version__ = "1.0.0"

from .diagnostics import run_diagnostic_suite, validate_runtime_invariants
__all__ = list(__all__) + ["run_diagnostic_suite", "validate_runtime_invariants"]
