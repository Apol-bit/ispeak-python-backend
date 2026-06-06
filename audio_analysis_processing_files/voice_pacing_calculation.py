# voice_pacing_calculation.py
from typing import List, Dict  # type hints for lists and dictionaries
import re  # regex for word extraction
import logging  # logging system

logger = logging.getLogger(__name__)  # create logger


def calculate_pacing(
    whisper_segments: List[Dict],  # list of speech segments
    audio_duration_seconds: float,  # total audio length
    slow_speech: int = 120,  # slow speech threshold (WPM)
    fast_speech: int = 150,  # fast speech threshold (WPM)
    pause_threshold: float = 1.0,  # minimum pause to count
    timing_jitter: float = 0.15,  # tolerance to ignore tiny gaps
) -> Dict:

    if audio_duration_seconds <= 0:
        raise ValueError("Invalid audio duration")  # prevent division errors

    # ---> NEW: Minimum audio length check for pacing <---
    # You cannot accurately judge someone's "pace" in less than 2 seconds.
    if audio_duration_seconds < 2.0:
        logger.info("Audio too short for accurate pacing calculation.")
        return {
            "wpm": 0.0,
            "articulation_rate": 0.0,
            "pacing_status": "Insufficient speech",
            "total_pause_seconds": 0.0,
        }

    if not whisper_segments:
        return {
            "wpm": 0.0,  # no words
            "articulation_rate": 0.0,  # no speaking rate
            "pacing_status": "No speech detected",  # message
            "total_pause_seconds": 0.0,  # no pauses
        }

    required_keys = {"start", "end", "text"}  # required fields

    # validate each segment
    for i, seg in enumerate(whisper_segments):
        if not required_keys <= seg.keys():
            raise ValueError(f"Malformed segment at index {i}")  # invalid data

    # ---------- WORD COUNT ----------
    total_words = 0  # total word count
    total_pause_time = 0.0  # total pause duration
    speaking_time = 0.0  # total speaking time

    for i, segment in enumerate(whisper_segments):

        # compute segment duration (avoid negative)
        duration = max(0.0, segment["end"] - segment["start"])
        speaking_time += duration  # add to speaking time

        # extract words using regex
        words = re.findall(r"[\w'-]+", segment["text"], re.UNICODE)
        total_words += len(words)  # count words

        # ---------- PAUSE DETECTION ----------
        if i > 0:
            previous = whisper_segments[i - 1]  # previous segment

            raw_pause = segment["start"] - previous["end"]  # gap between segments
            perceived_pause = max(0.0, raw_pause - timing_jitter)  # remove small noise

            # count pause if long enough
            if perceived_pause > pause_threshold:
                total_pause_time += perceived_pause

    if speaking_time <= 0:
        return {
            "wpm": 0.0,  # no valid speech
            "articulation_rate": 0.0,
            "pacing_status": "Insufficient speech",
            "total_pause_seconds": round(total_pause_time, 2),
        }

    # ---------- SPEED METRICS ----------
    real_minutes = audio_duration_seconds / 60.0  # total duration in minutes
    speaking_minutes = speaking_time / 60.0  # speaking time in minutes

    wpm = total_words / real_minutes  # words per minute (overall)
    articulation_rate = total_words / speaking_minutes  # words per speaking time

    # ---> NEW: Cap the absolute max WPM to prevent absurd numbers <---
    # The world record for fast talking is ~600 wpm. Normal is 150.
    wpm = min(wpm, 300.0)

    # ---------- PACING CLASSIFICATION ----------
    pacing_status = (
        "Slow pacing" if wpm < slow_speech  # below slow threshold
        else "Fast pacing" if wpm > fast_speech  # above fast threshold
        else "Excellent pacing"  # within range
    )

    # ---------- SANITY CHECK ----------
    if speaking_time > audio_duration_seconds * 1.05:
        logger.warning("Speaking time exceeds audio duration")  # possible error

    # return final results
    return {
        "wpm": round(wpm, 1),  # rounded WPM
        "articulation_rate": round(articulation_rate, 1),  # rounded articulation rate
        "pacing_status": pacing_status,  # pacing label
        "total_pause_seconds": round(total_pause_time, 2),  # total pauses
    }