from src.confidence.confidence import combined_confidence
from src.confidence.review_rules import review_required


def test_confidence_bounds():
    score = combined_confidence(1.0, 1.0, 1.0)
    assert score == 1.0
    assert review_required(0.2, 0.6)
