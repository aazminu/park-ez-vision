"""Deterministic runtime diagnostics and invariant verification for `park_ez`."""

from __future__ import annotations

import time
from typing import Any

from .occupancy import BoundingBox, compute_bbox_iou, polygon_area

def _domain_probe(seed: int) -> dict:
    b1 = BoundingBox(0.0, 0.0, 10.0, 10.0)
    b2 = BoundingBox(5.0, 5.0, 15.0, 15.0)
    iou = compute_bbox_iou(b1, b2)
    area = polygon_area([(0.0, 0.0), (10.0, 0.0), (10.0, 5.0), (0.0, 5.0)])
    return {
        "bbox_iou": round(iou, 4),
        "rect_area": round(area, 2),
        "invariants_ok": (abs(iou - (25.0 / 175.0)) < 1e-4 and area == 50.0),
    }


def run_diagnostic_suite(seed: int = 42) -> dict[str, Any]:
    """Execute deterministic domain probe for `park-ez-vision` and return structured metrics."""
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    t0 = time.perf_counter()
    metrics = _domain_probe(seed)
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)
    return {
        "repository": "park-ez-vision",
        "package": "park_ez",
        "seed": seed,
        "elapsed_ms": elapsed_ms,
        "metrics": metrics,
        "status": "ok" if metrics.get("invariants_ok", False) else "failed",
    }


def validate_runtime_invariants(seed: int = 42) -> dict[str, Any]:
    """Validate deterministic reproducibility and domain invariants for `park_ez`."""
    run_a = run_diagnostic_suite(seed=seed)
    run_b = run_diagnostic_suite(seed=seed)
    reproducible = run_a["metrics"] == run_b["metrics"]
    passed = bool(run_a["metrics"].get("invariants_ok", False)) and reproducible
    return {
        "repository": "park-ez-vision",
        "package": "park_ez",
        "seed": seed,
        "reproducible": reproducible,
        "passed": passed,
        "metrics": run_a["metrics"],
    }
