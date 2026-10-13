from __future__ import annotations


def anomaly_score(speed: float, confidence: float, expected_speed: float | None = None) -> float:
    """Transparent, non-learning anomaly proxy for the initial prototype.

    This is intentionally simple. A learned temporal anomaly model can replace it
    later without changing the state/event interfaces.
    """
    if expected_speed is None:
        return float(max(0.0, min(1.0, 1.0 - confidence)))
    deviation = abs(speed - expected_speed) / max(expected_speed, 1e-6)
    return float(max(0.0, min(1.0, 0.5 * deviation + 0.5 * (1.0 - confidence))))
