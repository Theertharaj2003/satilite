from __future__ import annotations


def combined_confidence(detection: float, identity: float = 1.0, context: float = 1.0) -> float:
    return float(max(0.0, min(1.0, 0.5 * detection + 0.3 * identity + 0.2 * context)))
