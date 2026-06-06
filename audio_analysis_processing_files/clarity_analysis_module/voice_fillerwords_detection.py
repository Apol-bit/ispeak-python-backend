# voice_fillerwords_detection.py
# ============================================================================
# Text-based filler word detection using hardcoded dictionaries.
#
# Detects filler words by scanning Whisper's transcription against
# comprehensive English and Filipino filler word dictionaries.
#
# Detection strategies:
#   1. Exact-match single-word fillers (always counted)
#   2. Multi-word filler phrases (e.g., "you know", "parang ganun")
#   3. Contextual fillers counted only when repeated (e.g., "like", "so")
# ============================================================================

from __future__ import annotations

import re
import logging
from typing import Any, Dict, List, Optional

import numpy as np  # kept for signature compatibility

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# ENGLISH filler words — always counted as fillers
# ---------------------------------------------------------------------------
_ENGLISH_ALWAYS_FILLER = {
    # Hesitation sounds
    "um", "umm", "ummm", "uhm", "uhmm",
    "uh", "uhh", "uhhh",
    "er", "err", "errr",
    "ah", "ahh", "ahhh",
    "em", "emm",
    "hm", "hmm", "hmmm",
    "mm", "mmm", "mmmm",
    "mhm", "aha",
    # Filler words
    "basically", "literally", "actually", "honestly",
    "anyway", "anyways", "whatever",
}

# ENGLISH multi-word filler phrases
_ENGLISH_MULTI_WORD_FILLERS = [
    "you know", "i mean", "you see", "kind of", "sort of",
    "i guess", "or something", "or whatever", "and stuff",
    "and everything", "and all that", "at the end of the day",
    "to be honest", "in a sense", "for the most part",
    "you know what i mean", "like i said", "as i said",
    "what i mean is", "the thing is", "i feel like",
]

# ENGLISH contextual fillers — only flagged when repeated within a window
_ENGLISH_REPEAT_ONLY_FILLERS = {
    "like", "so", "well", "right", "okay", "ok",
    "yeah", "yes", "just", "really", "very",
}

# ---------------------------------------------------------------------------
# FILIPINO (Tagalog) filler words — always counted as fillers
# ---------------------------------------------------------------------------
_FILIPINO_ALWAYS_FILLER = {
    # Common hesitation / filler sounds
    "ah", "ay", "eh", " eh", "oh",
    "ano", "anoh",                        # "what" used as filler
    "naman",                              # emphasis filler
    "kumbaga",                            # "so to speak"
    "parang",                             # "like / sort of"
    "ganun", "ganon",                     # "like that"
    "yung", "yun",                        # "that (thing)" — filler usage
    "basta",                              # "just because / anyway"
    "diba", "di ba",                      # "right?"
    "alam mo", "alam nyo",               # "you know"
    "nako", "hay nako",                   # exclamatory filler
    "kasi", "kase",                       # "because" used as filler
    "talaga",                             # "really" used as filler
    "siguro",                             # "maybe / i guess"
    "saka",                               # "and then" filler
    "tapos",                              # "then / and then" filler
    "yon", "yun",                         # "that's it" filler
    "pala",                               # "apparently" filler
    "nga",                                # emphasis filler
    "noh", "no",                          # "right?" tag filler
    "ba",                                 # question particle used as filler
    "hay", "hays",                        # sigh filler
    "aba",                                # exclamatory filler
    "sus", "jusko", "jusmio",            # exclamatory fillers
    "sige",                               # "okay / go ahead" filler
    "oo",                                 # "yes" used as filler
    "grabe",                              # "extreme" filler
    "totoo",                              # "true" used as filler
    "noh",                                # "right?" tag
    "ewan", "ewan ko",                   # "I don't know" filler
    "halimbawa",                          # "for example" used as filler padding
    "tsaka",                              # "and also" filler
}

# FILIPINO multi-word filler phrases
_FILIPINO_MULTI_WORD_FILLERS = [
    "alam mo", "alam nyo", "alam mo ba",
    "di ba", "diba",
    "hay nako", "naku naman",
    "parang ganun", "parang ganon",
    "kumbaga ganun", "kumbaga parang",
    "ano ba", "ano yun", "ano yung",
    "ewan ko", "ewan ko ba",
    "ang ibig sabihin", "ibig sabihin",
    "kasi naman", "kasi ano",
    "saka tapos", "tapos yung",
    "ganun ganun", "ganon ganon",
    "basta ganun", "basta ganon",
    "yung ano", "yung parang",
    "so ayun", "oo nga",
    "hay naku",
]

# FILIPINO contextual fillers — only flagged when repeated in a window
_FILIPINO_REPEAT_ONLY_FILLERS = {
    "tapos", "saka", "tsaka",
    "nga", "naman",
    "pala", "lang",
}

# ---------------------------------------------------------------------------
# Merged dictionaries
# ---------------------------------------------------------------------------
_ALL_ALWAYS_FILLER = _ENGLISH_ALWAYS_FILLER | _FILIPINO_ALWAYS_FILLER
_ALL_MULTI_WORD_FILLERS = _ENGLISH_MULTI_WORD_FILLERS + _FILIPINO_MULTI_WORD_FILLERS
_ALL_REPEAT_ONLY_FILLERS = _ENGLISH_REPEAT_ONLY_FILLERS | _FILIPINO_REPEAT_ONLY_FILLERS


def _clean_word(word: str) -> str:
    """Normalize a word for dictionary matching."""
    return re.sub(r"[^\w']", "", word).lower().strip("'")


# ---------------------------------------------------------------------------
# Text-based filler detection
# ---------------------------------------------------------------------------

def _detect_fillers_from_text(
    word_segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Scan Whisper word segments against the filler dictionaries.

    Three-pass detection:
      1. Exact-match always-filler words
      2. Contextual fillers (only when repeated within a 4-word window)
      3. Multi-word filler phrases
    """
    detected: List[Dict[str, Any]] = []
    seen_indices: set = set()
    cleaned_words = [_clean_word(seg.get("text", "")) for seg in word_segments]
    n = len(cleaned_words)

    # --- Pass 1: Always-filler single words ---
    for i, cw in enumerate(cleaned_words):
        if cw in _ALL_ALWAYS_FILLER:
            seen_indices.add(i)
            detected.append({
                "word": word_segments[i].get("text", "").strip(),
                "start": word_segments[i].get("start", 0.0),
                "end": word_segments[i].get("end", 0.0),
                "source": "text",
            })

    # --- Pass 2: Repeat-only fillers (flagged when repeated in a 4-word window) ---
    for i, cw in enumerate(cleaned_words):
        if i in seen_indices or cw not in _ALL_REPEAT_ONLY_FILLERS:
            continue
        window_end = min(i + 4, n)
        repeat_count = sum(1 for j in range(i, window_end) if cleaned_words[j] == cw)
        if repeat_count >= 2:
            for j in range(i, window_end):
                if cleaned_words[j] == cw and j not in seen_indices:
                    seen_indices.add(j)
                    detected.append({
                        "word": word_segments[j].get("text", "").strip(),
                        "start": word_segments[j].get("start", 0.0),
                        "end": word_segments[j].get("end", 0.0),
                        "source": "text",
                    })

    # --- Pass 3: Multi-word filler phrases ---
    for phrase in _ALL_MULTI_WORD_FILLERS:
        phrase_words = phrase.split()
        phrase_len = len(phrase_words)
        for i in range(n - phrase_len + 1):
            if all(cleaned_words[i + k] == phrase_words[k] for k in range(phrase_len)):
                any_new = any((i + k) not in seen_indices for k in range(phrase_len))
                if any_new:
                    phrase_text = " ".join(
                        word_segments[i + k].get("text", "").strip()
                        for k in range(phrase_len)
                    )
                    for k in range(phrase_len):
                        seen_indices.add(i + k)
                    detected.append({
                        "word": phrase_text,
                        "start": word_segments[i].get("start", 0.0),
                        "end": word_segments[i + phrase_len - 1].get("end", 0.0),
                        "source": "text",
                    })

    return detected


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def analyze_fillers(
    word_segments: List[Dict[str, Any]],
    y: Optional[np.ndarray] = None,
    sr: int = 16000,
    **kwargs,
) -> Dict[str, Any]:
    """
    Detect filler words by scanning the transcription against hardcoded
    English and Filipino filler word dictionaries.

    Args:
        word_segments: Word-level segments from Whisper [{text, start, end, confidence}]
        y:  Audio signal (unused — kept for backward compatibility).
        sr: Sample rate (unused — kept for backward compatibility).

    Returns:
        {
            "filler_count":  int,
            "filler_rate":   float,      # fillers per total words (%)
            "filler_words":  List[dict],
            "filler_score":  int,        # 0–100, 100 = no fillers
            "message":       str,
        }
    """
    words = [
        w.get("text", "").strip()
        for w in word_segments
        if w.get("text", "").strip()
    ]

    if not words:
        return {
            "filler_count": 0,
            "filler_rate":  0.0,
            "filler_words": [],
            "filler_score": 100,
            "message":      "No speech detected",
        }

    total_words = len(words)

    try:
        all_fillers = _detect_fillers_from_text(word_segments)
        logger.info("Dictionary scan found %d filler(s) from text", len(all_fillers))

    except Exception as e:
        logger.error("Filler detection failed: %s", e)
        return {
            "filler_count": 0,
            "filler_rate":  0.0,
            "filler_words": [],
            "filler_score": 100,
            "message":      "Analysis failed",
        }

    # Sort by time
    all_fillers.sort(key=lambda x: x.get("start", 0.0))

    # Deduplicate overlapping detections
    deduplicated: List[Dict[str, Any]] = []
    for filler in all_fillers:
        if not deduplicated:
            deduplicated.append(filler)
            continue
        prev = deduplicated[-1]
        if filler["start"] < prev.get("end", 0.0):
            continue
        deduplicated.append(filler)

    filler_count = len(deduplicated)
    filler_rate = round((filler_count / total_words) * 100, 1) if total_words > 0 else 0.0

    # Scoring: penalize based on filler rate
    if filler_rate <= 5.0:
        filler_score = round(100 - (filler_rate / 5.0) * 10)
    elif filler_rate <= 20.0:
        filler_score = round(90 - ((filler_rate - 5.0) / 15.0) * 40)
    else:
        filler_score = round(max(0, 50 - ((filler_rate - 20.0) / 20.0) * 50))

    # Message
    if filler_rate == 0:
        message = "No filler words detected"
    elif filler_rate < 5:
        message = "Minimal filler words"
    elif filler_rate < 15:
        message = "Moderate filler words — try to reduce hesitation sounds"
    else:
        message = "High filler word usage — practice speaking with fewer hesitations"

    logger.info(
        "Filler result: %d fillers / %d words (%.1f%%) → score %d — %s",
        filler_count, total_words, filler_rate, filler_score, message,
    )

    return {
        "filler_count": filler_count,
        "filler_rate":  filler_rate,
        "filler_words": deduplicated,
        "filler_score": filler_score,
        "message":      message,
    }