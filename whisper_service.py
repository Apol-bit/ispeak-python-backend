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
MIN_WORD_COUNT = 5

_SILENCE_RESPONSE = {
    "transcription": "",
    "word_timestamps": [],
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
        "score": 100,
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


def _extract_word_timestamps(segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract clean word-level timestamps for the teleprompter UI."""
    timestamps = []
    for seg in segments:
        for word in seg.get("words", []):
            w_text = word.get("word", "").strip()
            if w_text:
                timestamps.append({
                    "word": w_text,
                    "start": round(word.get("start", 0.0), 3),
                    "end": round(word.get("end", 0.0), 3),
                })
    return timestamps


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


def _run_analysis(file_path: str, y: np.ndarray, sr: int, model) -> Dict[str, Any]:
    """
    Core analysis logic shared by both standard and reference-based analysis.

    Returns the full analysis dict or the silence response.
    """

    # ---------- SILENCE GATE ----------
    rms = float(np.sqrt(np.mean(y ** 2)))
    logger.info("Audio RMS value: %f", rms)

    if rms < RMS_SILENCE_THRESHOLD:
        logger.info("RMS %.4f below silence threshold %.4f — skipping analysis.", rms, RMS_SILENCE_THRESHOLD)
        return dict(_SILENCE_RESPONSE)

    # ---------- NORMALIZE LOUDNESS ----------
    # Keep original for energy analysis (normalization destroys loudness info)
    y_original = y.copy()
    y = y * (0.05 / rms)

    # ---------- TRANSCRIBE ----------
    # Passing an initial_prompt with filler words strongly biases Whisper
    # against filtering out hesitation sounds like "um" and "uh" from the transcript.
    transcription = model.transcribe(
        file_path, 
        word_timestamps=True,
        initial_prompt="Umm, uh, hmm, like, you know, ah, ano, parang, yung."
    )

    text: str = transcription.get("text", "").strip()
    segments: List[Dict[str, Any]] = transcription.get("segments", [])

    # ---------- HALLUCINATION GUARD ----------
    word_count = len(text.split())
    if not text or word_count < MIN_WORD_COUNT:
        logger.info(
            "Transcript too short (%d word(s): %r) — treating as no speech.",
            word_count, text,
        )
        return dict(_SILENCE_RESPONSE)

    word_segments = _extract_word_segments(segments)
    if not word_segments:
        logger.info("No word-level timestamps returned by Whisper — treating as no speech.")
        return dict(_SILENCE_RESPONSE)

    # ---------- WORD TIMESTAMPS (for Teleprompter UI) ----------
    word_timestamps = _extract_word_timestamps(segments)

    # ---------- AUDIO DURATION ----------
    audio_duration = librosa.get_duration(y=y, sr=sr)

    # ---------- ENERGY ----------
    # Use ORIGINAL audio — normalization flattens loudness, making score always 100
    energy_stats = analyze_energy(y_original, sr)
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

    return {
        "transcription": text,
        "word_timestamps": word_timestamps,
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
            "words":   [
                {
                    "word":  fw.get("word", ""),
                    "start": round(fw.get("start", 0.0), 3),
                    "end":   round(fw.get("end", 0.0), 3),
                }
                for fw in filler_stats.get("filler_words", [])
            ],
            "message": filler_stats.get("message", ""),
        },
        # Internal — used by reference comparison
        "_internal": {
            "energy_stats": energy_stats,
            "pacing_stats": pacing_stats,
            "pacing_score": pacing_score,
            "clarity_score": clarity_score,
            "energy_score": energy_score,
        },
    }


def generate_full_analysis(file_path: str, model) -> Dict[str, Any]:
    """
    Run full speech analysis on an audio file (standard mode, no reference).

    Returns
    -------
    dict with keys:
        transcription, word_timestamps, scores, pacing, pronunciation, fillers
    """

    # ---------- LOAD AUDIO ----------
    y, sr = librosa.load(file_path, sr=16000, mono=True)

    result = _run_analysis(file_path, y, sr, model)

    # Remove internal data before returning
    result.pop("_internal", None)

    return result


def generate_reference_analysis(
    user_path: str,
    reference_path: str,
    model,
) -> Dict[str, Any]:
    """
    Run reference-based speech analysis.

    Compares the user's recorded audio against the Validator's reference audio.
    The reference audio's metrics serve as the "perfect score" baseline.
    Scores are adjusted based on how closely the user matches the reference.

    Returns
    -------
    dict — same structure as generate_full_analysis, with scores adjusted
           relative to the reference baseline.
    """

    logger.info("=== REFERENCE-BASED ANALYSIS ===")

    # ---------- LOAD BOTH AUDIO FILES ----------
    y_user, sr = librosa.load(user_path, sr=16000, mono=True)
    y_ref, sr_ref = librosa.load(reference_path, sr=16000, mono=True)

    # ---------- ANALYZE BOTH ----------
    logger.info("Analyzing reference audio...")
    ref_result = _run_analysis(reference_path, y_ref, sr_ref, model)

    logger.info("Analyzing user audio...")
    user_result = _run_analysis(user_path, y_user, sr, model)

    # If either analysis returned silence, return the user result as-is
    if not user_result.get("_internal") or not ref_result.get("_internal"):
        user_result.pop("_internal", None)
        return user_result

    ref_internal = ref_result["_internal"]
    user_internal = user_result["_internal"]

    # ---------- REFERENCE-BASED PACING SCORE ----------
    # Compare user WPM against reference WPM (instead of fixed 110-160 range)
    ref_wpm = ref_internal["pacing_stats"].get("wpm", 130.0) if isinstance(ref_internal["pacing_stats"], dict) else 130.0
    user_wpm = user_internal["pacing_stats"].get("wpm", 0.0) if isinstance(user_internal["pacing_stats"], dict) else 0.0

    if ref_wpm > 0 and user_wpm > 0:
        wpm_ratio = user_wpm / ref_wpm
        # Perfect ratio = 1.0 → score 100. Penalty for deviation.
        deviation = abs(1.0 - wpm_ratio)
        ref_pacing_score = round(max(0.0, 100.0 - (deviation * 100.0)), 1)
    else:
        ref_pacing_score = user_internal["pacing_score"]

    # ---------- REFERENCE-BASED ENERGY SCORE ----------
    # Compare energy profiles using RMS correlation
    ref_energy = ref_internal["energy_stats"]
    user_energy = user_internal["energy_stats"]

    # Use the user's energy score but adjust based on reference comparison
    ref_energy_score = user_internal["energy_score"]
    ref_is_monotone = ref_energy.get("is_monotone", False)
    user_is_monotone = user_energy.get("is_monotone", False)

    # If reference is monotone but user isn't, that's good — bonus
    if ref_is_monotone and not user_is_monotone:
        ref_energy_score = min(100.0, ref_energy_score + 10.0)
    # If user is monotone but reference isn't, that's bad — penalty already applied
    ref_energy_score = round(ref_energy_score, 1)

    # ---------- CLARITY stays the same (pronunciation is absolute, not relative) ----------
    ref_clarity_score = user_internal["clarity_score"]

    # ---------- RECALCULATE OVERALL ----------
    ref_overall = _compute_overall_score(ref_pacing_score, ref_clarity_score, ref_energy_score)

    # ---------- UPDATE USER RESULT ----------
    user_result["scores"]["pacing"] = ref_pacing_score
    user_result["scores"]["energy"] = ref_energy_score
    user_result["scores"]["overall"] = ref_overall

    # Clean up internal data
    user_result.pop("_internal", None)

    logger.info(
        "Reference comparison — Ref WPM: %.1f, User WPM: %.1f, "
        "Adj Pacing: %.1f, Adj Energy: %.1f, Adj Overall: %.1f",
        ref_wpm, user_wpm, ref_pacing_score, ref_energy_score, ref_overall,
    )

    return user_result