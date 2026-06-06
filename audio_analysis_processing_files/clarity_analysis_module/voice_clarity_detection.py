# voice_clarity_detection.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any
import statistics
import logging

logger = logging.getLogger(__name__)


@dataclass
class PronunciationConfig:
    minimum_confidence: float = 0.60   # minimum acceptable confidence
    minimum_word_duration: float = 0.15  # shortest expected word duration
    maximum_word_duration: float = 0.8   # longest expected word duration

    # NOTE: The ONNX Whisper model (phrase-level timestamps) always returns
    # probability=1.0 for every word, so confidence is not informative.
    # Duration analysis is the primary signal for pronunciation quality.
    confidence_weight: float = 0.0  # weight for confidence score
    duration_weight: float = 1.0    # weight for duration score

    # Threshold below which a duration score is flagged as "unusual".
    # Was 0.95 — far too strict, causing normal words to be flagged.
    # 0.75 only flags genuinely rushed/dragged words.
    duration_flag_threshold: float = 0.85


DEFAULT_CONFIG = PronunciationConfig()


SCORE_THRESHOLDS = [
    (85, "Excellent pronunciation"),
    (70, "Good pronunciation with minor issues"),
    (50, "Noticeable pronunciation problems"),
    (0,  "Pronunciation needs improvement"),
]


def _normalize_weights(cfg: PronunciationConfig) -> tuple[float, float]:
    total = cfg.confidence_weight + cfg.duration_weight
    if total <= 0:
        raise ValueError("Invalid scoring weights")
    return (
        cfg.confidence_weight / total,
        cfg.duration_weight / total,
    )


def _safe_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(confidence, 1.0))


def _duration_score(duration: float, cfg: PronunciationConfig) -> float:
    if duration <= 0:
        return 0.0

    if duration < cfg.minimum_word_duration:
        # Scale linearly: 0 duration → 0, at minimum → 1.0
        return duration / cfg.minimum_word_duration

    if duration > cfg.maximum_word_duration:
        excess_duration = duration - cfg.maximum_word_duration
        penalty_score = excess_duration / cfg.maximum_word_duration
        # Floor at 0.5 — even very long words aren't catastrophic
        return max(0.5, 1.0 - penalty_score)

    return 1.0  # within normal range → perfect score


def _feedback_message(score: int) -> str:
    return next(
        feedback for threshold, feedback in SCORE_THRESHOLDS
        if score >= threshold
    )


def analyze_pronunciation(
    word_segments: List[Dict[str, Any]],
    config: PronunciationConfig = DEFAULT_CONFIG,
) -> Dict[str, Any]:

    if not word_segments:
        return {
            "pronunciation_score": 0,
            "problematic_words": [],
            "message": "No speech detected",
        }

    conf_weight, dur_weight = _normalize_weights(config)

    scores: List[float] = []
    problematic_words: List[Dict[str, Any]] = []

    for word in word_segments:
        text = str(word.get("text", "")).strip()
        start = word.get("start")
        end = word.get("end")

        if not text:
            continue

        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            logger.warning("Invalid timestamps for word: %s", text)
            continue

        if end < start:
            logger.warning("Inverted timestamps for word: %s", text)
            duration = 0.0
            inverted = True
        else:
            duration = end - start
            inverted = False

        confidence = _safe_confidence(word.get("confidence"))
        dur_score = _duration_score(duration, config)

        word_score = conf_weight * confidence + dur_weight * dur_score
        scores.append(word_score)

        issues = []

        if confidence < config.minimum_confidence:
            issues.append("low confidence")

        # Fixed: was 0.95 — flagged almost every normal word as problematic.
        # Now uses configurable duration_flag_threshold (default 0.75).
        if dur_score < config.duration_flag_threshold:
            issues.append("unusual duration")

        if inverted:
            issues.append("invalid timestamps")

        if issues:
            problematic_words.append({
                "word": text,
                "confidence": round(confidence, 2),
                "duration": round(duration, 2),
                "issue": ", ".join(issues),
            })

    if not scores:
        return {
            "pronunciation_score": 0,
            "problematic_words": [],
            "message": "No valid words analyzed",
        }

    final_score = round(statistics.mean(scores) * 100)

    return {
        "pronunciation_score": final_score,
        "problematic_words": problematic_words,
        "message": _feedback_message(final_score),
    }