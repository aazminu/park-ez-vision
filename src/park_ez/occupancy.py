"""2D bounding-box and oriented polygon IoU geometry, 1D Kalman + temporal hysteresis
debounce filtering, and ADA van-accessible stall compliance monitoring.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned 2D bounding box in image or homography bird's-eye-view (BEV) coordinates."""

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float = 1.0
    has_ada_placard_tag: bool = False

    def __post_init__(self) -> None:
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError("BoundingBox max coordinates must be >= min coordinates.")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0].")

    @property
    def area(self) -> float:
        return max(0.0, self.x_max - self.x_min) * max(0.0, self.y_max - self.y_min)

    def to_polygon(self) -> tuple[tuple[float, float], ...]:
        return (
            (self.x_min, self.y_min),
            (self.x_max, self.y_min),
            (self.x_max, self.y_max),
            (self.x_min, self.y_max),
        )


@dataclass(frozen=True)
class ParkingStall:
    """Geometrically calibrated campus parking stall with optional ADA access aisle."""

    stall_id: str
    bbox: BoundingBox
    polygon: tuple[tuple[float, float], ...] | None = None
    is_ada_accessible: bool = False
    is_van_accessible: bool = False
    access_aisle_bbox: BoundingBox | None = None


@dataclass(frozen=True)
class StallTrackState:
    """Filtered occupancy state and ADA compliance status for a single parking stall."""

    stall_id: str
    kalman_occupancy_prob: float
    kalman_variance: float
    raw_max_iou: float
    consecutive_occupied_frames: int
    consecutive_vacant_frames: int
    is_occupied: bool
    is_ada_accessible: bool
    is_van_accessible: bool
    ada_violation: bool
    violation_reason: str | None


@dataclass(frozen=True)
class LotOccupancySummary:
    """Lot-level real-time availability and ADA violation telemetry."""

    total_stalls: int
    occupied_stalls: int
    vacant_regular_stalls: int
    vacant_ada_stalls: int
    vacant_van_stalls: int
    active_ada_violations: int
    violating_stall_ids: tuple[str, ...]


def compute_bbox_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Compute exact 2D Intersection-over-Union (IoU) between two axis-aligned boxes."""
    inter_x1 = max(box_a.x_min, box_b.x_min)
    inter_y1 = max(box_a.y_min, box_b.y_min)
    inter_x2 = min(box_a.x_max, box_b.x_max)
    inter_y2 = min(box_a.y_max, box_b.y_max)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area <= 0.0:
        return 0.0

    union_area = box_a.area + box_b.area - inter_area
    if union_area <= 1e-12:
        return 0.0
    return inter_area / union_area


def polygon_area(vertices: Sequence[tuple[float, float]]) -> float:
    """Compute unsigned area of a 2D polygon via the Shoelace formula."""
    n = len(vertices)
    if n < 3:
        return 0.0
    acc = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        acc += x1 * y2 - x2 * y1
    return 0.5 * abs(acc)


def _ensure_ccw(
    vertices: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Ensure polygon vertices are ordered counter-clockwise."""
    n = len(vertices)
    signed_area2 = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        signed_area2 += x1 * y2 - x2 * y1
    pts = list(vertices)
    if signed_area2 < 0:
        pts.reverse()
    return pts


def _sutherland_hodgman_clip(
    subject_polygon: Sequence[tuple[float, float]],
    clip_polygon: Sequence[tuple[float, float]],
) -> list[tuple[float, float]]:
    """Clip a convex subject polygon against a convex clip polygon via Sutherland-Hodgman."""
    if len(subject_polygon) < 3 or len(clip_polygon) < 3:
        return []

    output_list = _ensure_ccw(subject_polygon)
    clip_ccw = _ensure_ccw(clip_polygon)

    def _inside(
        p: tuple[float, float], cp1: tuple[float, float], cp2: tuple[float, float]
    ) -> bool:
        return (cp2[0] - cp1[0]) * (p[1] - cp1[1]) - (cp2[1] - cp1[1]) * (
            p[0] - cp1[0]
        ) >= -1e-9

    def _intersection(
        cp1: tuple[float, float],
        cp2: tuple[float, float],
        s: tuple[float, float],
        e: tuple[float, float],
    ) -> tuple[float, float]:
        dc = (cp1[0] - cp2[0], cp1[1] - cp2[1])
        dp = (s[0] - e[0], s[1] - e[1])
        n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
        n2 = s[0] * e[1] - s[1] * e[0]
        denom = dc[0] * dp[1] - dc[1] * dp[0]
        if abs(denom) < 1e-12:
            return e
        return ((n1 * dp[0] - n2 * dc[0]) / denom, (n1 * dp[1] - n2 * dc[1]) / denom)

    cp1 = clip_ccw[-1]
    for cp2 in clip_ccw:
        input_list = output_list
        output_list = []
        if not input_list:
            break
        s = input_list[-1]
        for e in input_list:
            if _inside(e, cp1, cp2):
                if not _inside(s, cp1, cp2):
                    output_list.append(_intersection(cp1, cp2, s, e))
                output_list.append(e)
            elif _inside(s, cp1, cp2):
                output_list.append(_intersection(cp1, cp2, s, e))
            s = e
        cp1 = cp2

    return output_list


def compute_polygon_iou(
    poly_a: Sequence[tuple[float, float]],
    poly_b: Sequence[tuple[float, float]],
) -> float:
    """Compute exact 2D Intersection-over-Union (IoU) between two convex polygons."""
    area_a = polygon_area(poly_a)
    area_b = polygon_area(poly_b)
    if area_a <= 0.0 or area_b <= 0.0:
        return 0.0

    inter_poly = _sutherland_hodgman_clip(poly_a, poly_b)
    inter_area = polygon_area(inter_poly)
    if inter_area <= 0.0:
        return 0.0

    union_area = area_a + area_b - inter_area
    if union_area <= 1e-12:
        return 0.0
    return max(0.0, min(1.0, inter_area / union_area))


class TemporalOccupancyTracker:
    """Multi-frame Kalman + hysteresis debounce occupancy filter with ADA stall guard."""

    def __init__(
        self,
        iou_threshold: float = 0.35,
        aisle_encroachment_iou_threshold: float = 0.08,
        enter_occupied_frames: int = 3,
        exit_vacant_frames: int = 3,
        process_noise_q: float = 0.02,
        measurement_noise_r: float = 0.08,
    ) -> None:
        if not (0.0 < iou_threshold < 1.0):
            raise ValueError("iou_threshold must be in (0, 1).")
        if enter_occupied_frames < 1 or exit_vacant_frames < 1:
            raise ValueError("Hysteresis frame thresholds must be >= 1.")
        self.iou_threshold = iou_threshold
        self.aisle_encroachment_iou_threshold = aisle_encroachment_iou_threshold
        self.enter_occupied_frames = enter_occupied_frames
        self.exit_vacant_frames = exit_vacant_frames
        self.process_noise_q = process_noise_q
        self.measurement_noise_r = measurement_noise_r
        self._states: dict[str, StallTrackState] = {}

    def get_state(self, stall_id: str) -> StallTrackState | None:
        return self._states.get(stall_id)

    def update_frame(
        self,
        stalls: Sequence[ParkingStall],
        detections: Sequence[BoundingBox],
    ) -> dict[str, StallTrackState]:
        """Process a single camera frame of vehicle bounding boxes against calibrated stalls."""
        for stall in stalls:
            prev = self._states.get(stall.stall_id)
            if prev is None:
                prev_prob = 0.05
                prev_var = 0.25
                occ_frames = 0
                vac_frames = 0
                is_occ = False
            else:
                prev_prob = prev.kalman_occupancy_prob
                prev_var = prev.kalman_variance
                occ_frames = prev.consecutive_occupied_frames
                vac_frames = prev.consecutive_vacant_frames
                is_occ = prev.is_occupied

            best_iou = 0.0
            best_det: BoundingBox | None = None
            for det in detections:
                if stall.polygon is not None:
                    iou = compute_polygon_iou(stall.polygon, det.to_polygon())
                else:
                    iou = compute_bbox_iou(stall.bbox, det)
                if iou > best_iou:
                    best_iou = iou
                    best_det = det

            raw_occupied_hit = best_iou >= self.iou_threshold
            measurement_z = (
                min(1.0, best_det.confidence * (best_iou / self.iou_threshold))
                if (raw_occupied_hit and best_det is not None)
                else 0.0
            )

            # 1D Kalman predict + update
            pred_prob = prev_prob
            pred_var = prev_var + self.process_noise_q
            kalman_gain = pred_var / (pred_var + self.measurement_noise_r)
            post_prob = max(
                0.0, min(1.0, pred_prob + kalman_gain * (measurement_z - pred_prob))
            )
            post_var = (1.0 - kalman_gain) * pred_var

            # Multi-frame hysteresis debounce state machine
            if raw_occupied_hit:
                occ_frames += 1
                vac_frames = 0
                if not is_occ and occ_frames >= self.enter_occupied_frames:
                    is_occ = True
            else:
                vac_frames += 1
                occ_frames = 0
                if is_occ and vac_frames >= self.exit_vacant_frames:
                    is_occ = False

            # ADA accessible stall & van access aisle compliance logic
            ada_violation = False
            violation_reason: str | None = None

            if stall.access_aisle_bbox is not None:
                for det in detections:
                    aisle_iou = compute_bbox_iou(stall.access_aisle_bbox, det)
                    if aisle_iou >= self.aisle_encroachment_iou_threshold:
                        ada_violation = True
                        violation_reason = "ACCESS_AISLE_ENCROACHMENT"
                        break

            if (
                not ada_violation
                and stall.is_ada_accessible
                and is_occ
                and best_det is not None
                and not best_det.has_ada_placard_tag
            ):
                ada_violation = True
                violation_reason = "UNAUTHORIZED_NO_ADA_PERMIT"

            self._states[stall.stall_id] = StallTrackState(
                stall_id=stall.stall_id,
                kalman_occupancy_prob=post_prob,
                kalman_variance=post_var,
                raw_max_iou=best_iou,
                consecutive_occupied_frames=occ_frames,
                consecutive_vacant_frames=vac_frames,
                is_occupied=is_occ,
                is_ada_accessible=stall.is_ada_accessible,
                is_van_accessible=stall.is_van_accessible,
                ada_violation=ada_violation,
                violation_reason=violation_reason,
            )

        return dict(self._states)

    def summarize_lot(self, stalls: Sequence[ParkingStall]) -> LotOccupancySummary:
        """Aggregate lot-level occupancy counts and active ADA violations."""
        total = len(stalls)
        occupied = 0
        vacant_reg = 0
        vacant_ada = 0
        vacant_van = 0
        violating_ids: list[str] = []

        for stall in stalls:
            state = self._states.get(stall.stall_id)
            is_occ = state.is_occupied if state is not None else False
            has_viol = state.ada_violation if state is not None else False

            if has_viol:
                violating_ids.append(stall.stall_id)

            if is_occ:
                occupied += 1
            else:
                if stall.is_ada_accessible:
                    vacant_ada += 1
                    if stall.is_van_accessible:
                        vacant_van += 1
                else:
                    vacant_reg += 1

        return LotOccupancySummary(
            total_stalls=total,
            occupied_stalls=occupied,
            vacant_regular_stalls=vacant_reg,
            vacant_ada_stalls=vacant_ada,
            vacant_van_stalls=vacant_van,
            active_ada_violations=len(violating_ids),
            violating_stall_ids=tuple(violating_ids),
        )
