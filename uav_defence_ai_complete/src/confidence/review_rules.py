from __future__ import annotations


def review_required(confidence: float, threshold: float = 0.60) -> bool:
    return confidence < threshold


def review_reason(confidence: float, detection: float, identity: float, context: float, threshold: float = 0.60) -> list[str]:
    reasons: list[str] = []
    if detection < threshold:
        reasons.append("low_detection_confidence")
    if identity < threshold:
        reasons.append("low_identity_confidence")
    if context < threshold:
        reasons.append("low_context_confidence")
    if confidence < threshold and not reasons:
        reasons.append("combined_confidence_below_threshold")
    return reasons
