# filler_detector.py
from __future__ import annotations

import logging
from typing import Any, Dict, List
from transformers import pipeline

logger = logging.getLogger(__name__)

_classifier = None  # lazy-loaded singleton


def _get_classifier(model_path: str):
    global _classifier
    if _classifier is None:
        _classifier = pipeline(
            "token-classification",
            model=model_path,
            tokenizer=model_path,
            aggregation_strategy="simple",
        )
        logger.info("Filler classifier loaded from %s", model_path)
    return _classifier


def analyze_fillers(
    word_segments: List[Dict[str, Any]],
    model_path: str = "models/filler_model_best",
    confidence_threshold: float = 0.50, # LOWERED THRESHOLD (Was 0.70)
) -> Dict[str, Any]:
    """
    Detect filler words from a list of word segments.

    Uses the same input format as analyze_pronunciation() for consistency.

    Returns:
        {
            "filler_count":  int,
            "filler_rate":   float,  # fillers per 100 words
            "filler_words":  List[dict],
            "filler_score":  int,    # 0–100, 100 = no fillers
            "message":       str,
        }
    """
    words = [
        w.get("text", "").strip()
        for w in word_segments
        if w.get("text", "").strip()
    ]

    # No speech → no fillers → perfect filler score.
    # Was returning filler_score=0 here, which incorrectly dragged clarity down.
    if not words:
        return {
            "filler_count": 0,
            "filler_rate":  0.0,
            "filler_words": [],
            "filler_score": 100,
            "message":      "No speech detected",
        }

    sentence = " ".join(words)

    try:
        clf = _get_classifier(model_path)
        predictions = clf(sentence)
    except Exception as e:
        logger.error("Filler detection failed: %s", e)
        # Analysis failed → do not penalize the speaker; return neutral score.
        # Was returning filler_score=0, unfairly tanking clarity on model errors.
        return {
            "filler_count": 0,
            "filler_rate":  0.0,
            "filler_words": [],
            "filler_score": 100,
            "message":      "Analysis failed",
        }

    flagged = [
        {"word": p["word"].strip(), "confidence": round(p["score"], 3)}
        for p in predictions
        if p["entity_group"] == "FILLER" and p["score"] > confidence_threshold
    ]

    total_words  = len(words)
    filler_count = len(flagged)
    filler_rate  = round((filler_count / total_words) * 100, 1)

    # Score formula:
    # Old: 100 - (filler_rate / 30) * 100
    #   → at 10% fillers score was already ~67, too punishing for moderate use.
    # New: two-stage curve
    #   - 0–5%   fillers: score stays near 100 (minor hesitations are natural)
    #   - 5–20%  fillers: linear drop from 100 → 50
    #   - 20%+   fillers: linear drop from 50 → 0 (severe)
    if filler_rate <= 5.0:
        filler_score = round(100 - (filler_rate / 5.0) * 10)   # 100 → 90
    elif filler_rate <= 20.0:
        filler_score = round(90 - ((filler_rate - 5.0) / 15.0) * 40)  # 90 → 50
    else:
        filler_score = round(max(0, 50 - ((filler_rate - 20.0) / 20.0) * 50))  # 50 → 0

    if filler_rate == 0:
        message = "No filler words detected"
    elif filler_rate < 5:
        message = "Minimal filler words"
    elif filler_rate < 15:
        message = "Moderate filler words — try to reduce hesitation sounds"
    else:
        message = "High filler word usage — practice speaking with fewer hesitations"

    return {
        "filler_count": filler_count,
        "filler_rate":  filler_rate,
        "filler_words": flagged,
        "filler_score": filler_score,
        "message":      message,
    }