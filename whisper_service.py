from __future__ import annotations

import numpy as np
import logging
import librosa
from typing import Any, Dict, List

from audio_analysis_processing_files.voice_energy_analyze import analyze_energy
from audio_analysis_processing_files.voice_pacing_calculation import calculate_pacing
from audio_analysis_processing_files.clarity_analysis_module.voice_clarity_detection import analyze_pronunciation
from audio_analysis_processing_files.clarity_analysis_module.voice_fillerwords_detection import analyze_fillers

logger = logging.getLogger(__name__)

# Minimum RMS to treat audio as containing real speech.
# 1e-4 was far too low — background noise easily exceeds it.
RMS_SILENCE_THRESHOLD = 0.01

# Minimum real words required after transcription.
# Guards against Whisper hallucinating text on noise.
MIN_WORD_COUNT = 3

_SILENCE_RESPONSE = {
    "transcription": "",
    "scores": {
        "overall": 0,
        "clarity": 0,
        "pacing": 0,
        "energy": 0,
    },
    "pacing": {
        "wpm": 0.0,
        "message": "No speech detected"
    },
    "pronunciation": {
        "score": 0,
        "message": "No speech detected",
        "problematic_words": [],
    },
    "fillers": {
        "score": 0,
        "count": 0,
        "rate": 0.0,
        "words": [],
        "message": "No speech detected",
    },
}


def _extract_word_segments(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten and remap Whisper word segments for pronunciation analysis."""
    return [
        {
            "text": word.get("word", ""),
            "start": word.get("start"),
            "end": word.get("end"),
            "confidence": word.get("probability", 0.0),
        }
        for seg in segments
        for word in seg.get("words", [])
    ]


def _compute_pacing_score(pacing_stats: Dict[str, Any]) -> float:
    """
    Convert pacing status into a 0-100 score.

    Excellent pacing → 100
    Slow or fast     → scaled by how far WPM is from the ideal range
    """
    status = pacing_stats.get("pacing_status", "")
    wpm = pacing_stats.get("wpm", 0.0)

    if status in ("No speech detected", "Insufficient speech", "Analysis failed"):
        return 0.0

    if status == "Excellent pacing":
        return 100.0

    # Ideal range is 110-160 WPM (from calculate_pacing defaults)
    if status == "Slow pacing":
        # 0 WPM → 0, approaching 110 WPM → 100
        return round(min(100.0, max(0.0, (wpm / 110.0) * 100)), 1)

    if status == "Fast pacing":
        # 160 WPM → 100, penalty grows beyond that
        excess = wpm - 160.0
        return round(min(100.0, max(0.0, 100.0 - (excess / 160.0) * 100)), 1)

    # fallback (e.g. "Analysis failed", "No speech detected")
    return 0.0


def _compute_energy_score(energy_stats: Dict[str, Any]) -> float:
    """
    Convert energy stats into a 0-100 score.

    Factors:
    - loudness_status: Normal volume → full points
    - is_low_variation: penalize monotone energy
    - is_monotone: penalize monotone pitch
    """
    score = 100.0

    loudness_status = energy_stats.get("loudness_status", "")
    if loudness_status == "Too quiet (whispering)":
        score -= 40.0
    elif loudness_status == "Too loud (shouting)":
        score -= 30.0
    elif loudness_status == "Silence":
        return 0.0

    if energy_stats.get("is_low_variation"):
        score -= 20.0

    if energy_stats.get("is_monotone"):
        score -= 20.0

    return round(max(0.0, score), 1)


def _compute_overall_score(
    pacing_score: float,
    clarity_score: float,
    energy_score: float,
) -> float:
    """
    Weighted average of all three scores.

    Weights:
    - Clarity     40%  (most important for speech quality)
    - Pacing      35%
    - Energy      25%
    """
    overall = (
        clarity_score * 0.40 +
        pacing_score  * 0.35 +
        energy_score  * 0.25
    )
    return round(overall, 1)


def generate_full_analysis(file_path: str, model) -> Dict[str, Any]:
    """
    Run full speech analysis on an audio file.

    Returns
    -------
    dict with keys:
        transcription, pacing_score, clarity_score, energy_score, overall_score
    """

    # ---------- LOAD AUDIO ----------
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    # ---------- SILENCE GATE ----------
    # Raised from 1e-4 to 0.01 — background noise easily clears the old threshold.
    rms = float(np.sqrt(np.mean(y ** 2)))
    logger.info("Audio RMS value: %f", rms)

    if rms < RMS_SILENCE_THRESHOLD:
        logger.info("RMS %.4f below silence threshold %.4f — skipping analysis.", rms, RMS_SILENCE_THRESHOLD)
        return dict(_SILENCE_RESPONSE)

    # ---------- NORMALIZE LOUDNESS ----------
    # Only normalize after the silence gate so we don't amplify pure noise
    # into something that fools Whisper and the energy scorer.
    y = y * (0.05 / rms)

    # ---------- TRANSCRIBE ----------
    transcription = model.transcribe(file_path, word_timestamps=True)

    text: str = transcription.get("text", "").strip()
    segments: List[Dict[str, Any]] = transcription.get("segments", [])

    # ---------- HALLUCINATION GUARD ----------
    # Whisper is known to generate phantom words on noise/silence.
    # Reject if the transcript is empty or suspiciously short.
    word_count = len(text.split())
    if not text or word_count < MIN_WORD_COUNT:
        logger.info(
            "Transcript too short (%d word(s): %r) — treating as no speech.",
            word_count, text,
        )
        return dict(_SILENCE_RESPONSE)

    # Also reject if Whisper returned text but produced no word-level timestamps,
    # which is another hallucination signal.
    word_segments = _extract_word_segments(segments)
    if not word_segments:
        logger.info("No word-level timestamps returned by Whisper — treating as no speech.")
        return dict(_SILENCE_RESPONSE)

    # ---------- AUDIO DURATION ----------
    audio_duration = transcription.get("duration")
    if not audio_duration:
        audio_duration = librosa.get_duration(y=y, sr=sr)
        logger.warning(
            "Duration missing from Whisper result — "
            "falling back to librosa: %.2fs", audio_duration
        )

    # ---------- ENERGY ----------
    energy_stats = analyze_energy(y, sr)
    energy_score = _compute_energy_score(energy_stats)

    # ---------- PACING ----------
    try:
        pacing_stats = calculate_pacing(segments, audio_duration)
        pacing_score = _compute_pacing_score(pacing_stats)
    except ValueError as e:
        logger.error("Pacing analysis failed: %s", e)
        pacing_stats = {}
        pacing_score = 0.0

    # ---------- PRONUNCIATION ----------
    pronunciation_stats = analyze_pronunciation(word_segments)
    pronunciation_score = float(pronunciation_stats.get("pronunciation_score", 0))

    # ---------- FILLER ----------
    filler_stats = analyze_fillers(word_segments)
    filler_score = float(filler_stats.get("filler_score", 100))

    # ---------- CLARITY ----------
    clarity_score = round(
        pronunciation_score * 0.60 +
        filler_score        * 0.40
    )

    # ---------- OVERALL ----------
    overall_score = _compute_overall_score(pacing_score, clarity_score, energy_score)

    # ---------- RESULT ----------
    return {
        "transcription": text,
        "scores": {
            "overall":    overall_score,
            "clarity":    clarity_score,
            "pacing":     pacing_score,
            "energy":     energy_score,
        },
        "pacing": {
            "wpm": pacing_stats.get("wpm", 0.0) if isinstance(pacing_stats, dict) else 0.0,
            "message": pacing_stats.get("pacing_status", "") if isinstance(pacing_stats, dict) else "Analysis failed"
        },
        "pronunciation": {
            "score":             pronunciation_score,
            "message":           pronunciation_stats.get("message", ""),
            "problematic_words": [
                {
                    "word":       w["word"],
                    "confidence": w["confidence"],
                    "duration":   float(w["duration"]),
                    "issue":      w["issue"],
                }
                for w in pronunciation_stats.get("problematic_words", [])
            ],
        },
        "fillers": {
            "score":   filler_score,
            "count":   filler_stats.get("filler_count", 0),
            "rate":    filler_stats.get("filler_rate", 0.0),
            "words":   filler_stats.get("filler_words", []),
            "message": filler_stats.get("message", ""),
        },
    }