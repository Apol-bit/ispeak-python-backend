# voice_clarity_detection.py
from __future__ import annotations  # allows modern type hints (like tuple[int, int]) in older Python

from dataclasses import dataclass  # for creating simple config class
from typing import List, Dict, Any  # type hints
import statistics  # for mean calculation
import logging  # for logging warnings

logger = logging.getLogger(__name__)  # create logger for this module


# Config class for tuning scoring behavior
@dataclass
class PronunciationConfig:
    minimum_confidence: float = 0.75  # minimum acceptable confidence
    minimum_word_duration: float = 0.08  # shortest expected word duration
    maximum_word_duration: float = 1.2  # longest expected word duration

    confidence_weight: float = 0.7  # weight for confidence score
    duration_weight: float = 0.3  # weight for duration score


DEFAULT_CONFIG = PronunciationConfig()  # default config instance


# Score thresholds → feedback message
SCORE_THRESHOLDS = [
    (85, "Excellent pronunciation"),
    (70, "Good pronunciation with minor issues"),
    (50, "Noticeable pronunciation problems"),
    (0,  "Pronunciation needs improvement"),
]


# Normalize weights so they sum to 1
def _normalize_weights(cfg: PronunciationConfig) -> tuple[float, float]:
    total = cfg.confidence_weight + cfg.duration_weight  # sum weights

    if total <= 0:
        raise ValueError("Invalid scoring weights")  # prevent divide by zero

    return (
        cfg.confidence_weight / total,  # normalized confidence weight
        cfg.duration_weight / total,  # normalized duration weight
    )


# Safely convert confidence to float and clamp to [0,1]
def _safe_confidence(value: Any) -> float:
    try:
        confidence = float(value)  # try converting
    except (TypeError, ValueError):
        return 0.0  # fallback if invalid

    return max(0.0, min(confidence, 1.0))  # clamp between 0 and 1


# Compute duration score (0–1). How fast the word is spoken.
def _duration_score(duration: float, cfg: PronunciationConfig) -> float:

    if duration <= 0:
        return 0.0  # invalid duration

    if duration < cfg.minimum_word_duration:
        return duration / cfg.minimum_word_duration  # scale up short words

    if duration > cfg.maximum_word_duration:
        excess_duration = duration - cfg.maximum_word_duration  # extra time
        penalty_score = excess_duration / cfg.maximum_word_duration  # penalty ratio
        return max(0.5, 1.0 - penalty_score)  # reduce score but not below 0.5

    return 1.0  # perfect duration


# Get feedback message based on score
def _feedback_message(score: int) -> str:
    return next(
        feedback for threshold, feedback in SCORE_THRESHOLDS
        if score >= threshold
    )


# Main function
def analyze_pronunciation(
    word_segments: List[Dict[str, Any]],
    config: PronunciationConfig = DEFAULT_CONFIG
) -> Dict[str, Any]:

    # Handle empty input
    if not word_segments:
        return {
            "pronunciation_score": 0,
            "problematic_words": [],
            "message": "No speech detected"
        }

    conf_weight, dur_weight = _normalize_weights(config)  # get normalized weights

    scores: List[float] = []  # store word scores
    problematic_words: List[Dict[str, Any]] = []  # store issues

    for word in word_segments:

        text = str(word.get("text", "")).strip()  # get word text
        start = word.get("start")  # start time
        end = word.get("end")  # end time

        if not text:
            continue  # skip empty words

        # Validate timestamps
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            logger.warning("Invalid timestamps for word: %s", text)
            continue

        if end < start:
            logger.warning("Inverted timestamps for word: %s", text)
            duration = 0.0  # invalid duration
            inverted = True
        else:
            duration = end - start  # compute duration
            inverted = False

        confidence = _safe_confidence(word.get("confidence"))  # clean confidence

        dur_score = _duration_score(duration, config)  # compute duration score

        # Final word score (weighted)
        word_score = (
            conf_weight * confidence +
            dur_weight * dur_score
        )

        scores.append(word_score)  # store score

        # Detect issues
        issues = []

        if confidence < config.minimum_confidence:
            issues.append("low confidence")

        if dur_score < 0.95:
            issues.append("unusual duration")

        if inverted:
            issues.append("invalid timestamps")

        # Save problematic words
        if issues:
            problematic_words.append({
                "word": text,
                "confidence": round(confidence, 2),
                "duration": round(duration, 2),
                "issue": ", ".join(issues),
            })

    # No valid scores
    if not scores:
        return {
            "pronunciation_score": 0,
            "problematic_words": [],
            "message": "No valid words analyzed"
        }

    final_score = round(statistics.mean(scores) * 100)  # average → percentage

    return {
        "pronunciation_score": final_score,
        "problematic_words": problematic_words,
        "message": _feedback_message(final_score),  # final feedback
    }