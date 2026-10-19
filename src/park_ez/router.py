"""A* campus road network routing and multi-criteria accessible parking lot recommender."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Sequence


@dataclass(frozen=True)
class CampusNode:
    """Intersection or parking lot entrance node on the SMU campus road network."""

    node_id: str
    xy: tuple[float, float]


@dataclass(frozen=True)
class RoadEdge:
    """Directed road segment with length (m) and free-flow or live speed (m/s)."""

    to_node: str
    distance_m: float
    speed_limit_mps: float = 11.1

    def __post_init__(self) -> None:
        if self.distance_m <= 0:
            raise ValueError("distance_m must be positive.")
        if self.speed_limit_mps <= 0:
            raise ValueError("speed_limit_mps must be positive.")


@dataclass(frozen=True)
class ParkingLotCandidate:
    """Campus parking structure or surface lot with live occupancy and pedestrian grade metrics."""

    lot_id: str
    node_id: str
    xy: tuple[float, float]
    total_stalls: int
    vacant_regular_stalls: int
    vacant_ada_stalls: int
    vacant_van_stalls: int
    hourly_turnover_rate: float
    walk_slope_pct_to_dest: float
    has_curb_cut_ramp: bool = True


@dataclass(frozen=True)
class LotRecommendation:
    """Ranked parking lot recommendation with A* route, walking accessibility, and stall capture probability."""

    lot_id: str
    route_nodes: list[str]
    driving_distance_m: float
    driving_eta_seconds: float
    walking_distance_m: float
    walking_grade_pct: float
    availability_probability: float
    composite_cost: float


def _euclidean_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def shortest_driving_path_astar(
    nodes: dict[str, CampusNode],
    adjacency: dict[str, Sequence[RoadEdge]],
    start_node: str,
    goal_node: str,
    max_speed_mps: float = 15.0,
) -> tuple[list[str], float, float]:
    """Compute fastest driving route on the campus road graph via A* search.

    Returns:
        Tuple of (route_node_ids, total_distance_m, driving_eta_seconds).
    """
    if start_node not in nodes or goal_node not in nodes:
        raise KeyError("start_node and goal_node must exist in nodes.")
    if max_speed_mps <= 0:
        raise ValueError("max_speed_mps must be positive.")

    goal_xy = nodes[goal_node].xy

    def _heuristic_seconds(nid: str) -> float:
        return _euclidean_m(nodes[nid].xy, goal_xy) / max_speed_mps

    open_heap: list[tuple[float, float, float, str, list[str]]] = [
        (_heuristic_seconds(start_node), 0.0, 0.0, start_node, [start_node])
    ]
    best_g: dict[str, float] = {start_node: 0.0}

    while open_heap:
        _, g_time, g_dist, curr, path = heapq.heappop(open_heap)
        if curr == goal_node:
            return path, g_dist, g_time

        if g_time > best_g.get(curr, float("inf")) + 1e-9:
            continue

        for edge in adjacency.get(curr, ()):
            edge_time = edge.distance_m / edge.speed_limit_mps
            tentative_g = g_time + edge_time
            if tentative_g < best_g.get(edge.to_node, float("inf")):
                best_g[edge.to_node] = tentative_g
                f_score = tentative_g + _heuristic_seconds(edge.to_node)
                heapq.heappush(
                    open_heap,
                    (
                        f_score,
                        tentative_g,
                        g_dist + edge.distance_m,
                        edge.to_node,
                        path + [edge.to_node],
                    ),
                )

    raise ValueError(f"No directed driving path from {start_node} to {goal_node}.")


def estimate_stall_capture_probability(
    vacant_stalls: int,
    total_stalls: int,
    hourly_turnover_rate: float,
    driving_eta_seconds: float,
) -> float:
    """Estimate probability of securing an open stall upon arrival after driving_eta_seconds.

    Uses a continuous-time birth-death / Poisson turnover model that accounts for current
    open capacity and expected stall churn during transit.
    """
    if total_stalls <= 0:
        return 0.0
    vacant = max(0, vacant_stalls)
    eta_hours = max(0.0, driving_eta_seconds) / 3600.0
    decay = math.exp(-1.5 * eta_hours)
    turnover_expected = max(0.0, hourly_turnover_rate) * eta_hours

    effective_open = vacant * decay + 0.35 * turnover_expected
    if effective_open <= 0.0:
        return 0.0
    prob = 1.0 - math.exp(-1.25 * effective_open)
    return max(0.0, min(0.999, prob))


def recommend_parking_lots(
    nodes: dict[str, CampusNode],
    adjacency: dict[str, Sequence[RoadEdge]],
    lots: Sequence[ParkingLotCandidate],
    origin_node: str,
    target_building_xy: tuple[float, float],
    requires_ada: bool = False,
    requires_van_accessible: bool = False,
    max_ada_walk_slope_pct: float = 5.0,
    walking_speed_mps: float = 1.35,
) -> list[LotRecommendation]:
    """Rank campus parking lots by minimizing composite driving, accessible walking, and turnover cost.

    When `requires_ada` or `requires_van_accessible` is enabled:
      - Filters out lots lacking curb-cut ramps (`has_curb_cut_ramp == False`)
      - Filters out lots whose pedestrian pathway exceeds `max_ada_walk_slope_pct`
        (default 5.0% per ADA Standards §403.3 running slope limit)
      - Evaluates availability strictly on `vacant_van_stalls` or `vacant_ada_stalls`
    """
    recommendations: list[LotRecommendation] = []

    for lot in lots:
        if requires_van_accessible:
            relevant_vacant = lot.vacant_van_stalls
        elif requires_ada:
            relevant_vacant = lot.vacant_ada_stalls
        else:
            relevant_vacant = lot.vacant_regular_stalls

        if (requires_ada or requires_van_accessible) and not lot.has_curb_cut_ramp:
            continue
        if (
            requires_ada or requires_van_accessible
        ) and lot.walk_slope_pct_to_dest > max_ada_walk_slope_pct:
            continue
        if relevant_vacant <= 0 and lot.hourly_turnover_rate <= 0.0:
            continue

        try:
            route_nodes, drive_dist_m, drive_eta_s = shortest_driving_path_astar(
                nodes=nodes,
                adjacency=adjacency,
                start_node=origin_node,
                goal_node=lot.node_id,
            )
        except ValueError:
            continue

        walk_dist_m = _euclidean_m(lot.xy, target_building_xy)
        walk_time_s = walk_dist_m / max(0.5, walking_speed_mps)

        capture_prob = estimate_stall_capture_probability(
            vacant_stalls=relevant_vacant,
            total_stalls=lot.total_stalls,
            hourly_turnover_rate=lot.hourly_turnover_rate,
            driving_eta_seconds=drive_eta_s,
        )
        if capture_prob <= 1e-4:
            continue

        # Grade penalty increases superlinearly with slope, amplified for ADA users
        grade_weight = 45.0 if (requires_ada or requires_van_accessible) else 12.0
        grade_penalty = grade_weight * (max(0.0, lot.walk_slope_pct_to_dest) ** 1.4)

        # Scarcity penalty based on log-odds of failing to find a stall upon arrival
        scarcity_penalty = 180.0 * (1.0 - capture_prob) / max(0.05, capture_prob)

        walk_weight = 2.4 if (requires_ada or requires_van_accessible) else 1.5
        composite_cost = (
            1.0 * drive_eta_s
            + walk_weight * walk_time_s
            + grade_penalty
            + scarcity_penalty
        )

        recommendations.append(
            LotRecommendation(
                lot_id=lot.lot_id,
                route_nodes=route_nodes,
                driving_distance_m=drive_dist_m,
                driving_eta_seconds=drive_eta_s,
                walking_distance_m=walk_dist_m,
                walking_grade_pct=lot.walk_slope_pct_to_dest,
                availability_probability=capture_prob,
                composite_cost=composite_cost,
            )
        )

    recommendations.sort(key=lambda r: r.composite_cost)
    return recommendations
