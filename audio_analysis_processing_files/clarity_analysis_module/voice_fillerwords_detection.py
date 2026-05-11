# filler_detector.py
from __future__ import annotations

import logging
from typing import Any, Dict, List
from transformers import pipeline

logger = logging.getLogger(__name__)

_classifier = None   # lazy-loaded singleton

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
    confidence_threshold: float = 0.70,
) -> Dict[str, Any]:
    """
    Detect filler words from a list of word segments.

    Uses the same input format as analyze_pronunciation() for consistency.

    Returns:
        {
            "filler_count":      int,
            "filler_rate":       float,   # fillers per 100 words
            "filler_words":      List[dict],
            "filler_score":      int,     # 0-100, 100 = no fillers
            "message":           str,
        }
    """
    words = [w.get("text", "").strip() for w in word_segments if w.get("text", "").strip()]
    if not words:
        return {
            "filler_count": 0,
            "filler_rate": 0.0,
            "filler_words": [],
            "filler_score": 0,      
            "message": "No speech detected",
        }

    sentence = " ".join(words)

    try:
        clf = _get_classifier(model_path)
        predictions = clf(sentence)
    except Exception as e:
        logger.error("Filler detection failed: %s", e)
        return {
            "filler_count": 0,
            "filler_rate": 0.0,
            "filler_words": [],
            "filler_score": 0,       
            "message": "Analysis failed",
        }

    flagged = [
        {"word": p["word"].strip(), "confidence": round(p["score"], 3)}
        for p in predictions
        if p["entity_group"] == "FILLER" and p["score"] > confidence_threshold
    ]

    total_words  = len(words)
    filler_count = len(flagged)
    filler_rate  = round((filler_count / total_words) * 100, 1)

    # Score: 100 at 0 fillers, drops toward 0 at ~30% filler rate
    filler_score = max(0, round(100 - (filler_rate / 30) * 100))

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