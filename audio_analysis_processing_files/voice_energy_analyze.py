# voice_energy_analyze.py
import numpy as np  # numerical operations
import librosa  # audio processing library
from typing import Dict, List  # type hints


# ---------------- CONFIG ----------------

WHISPER_THRESHOLD_DB = -20.0  # below this = whispering
SHOUT_THRESHOLD_DB = -8.0  # above this = shouting

LOW_VARIATION_DB = 10.0  # variation < 10 dB → speech is flat / dull
MONOTONE_PITCH_STD_THRESHOLD = 25.0  # pitch std < 25 Hz → monotone

MIN_ACTIVE_RATIO = 0.10  # at least 10% of frames must be non-silent
MIN_FRAMES_FOR_ROBUST_PERCENTILE = 50  # min frames for stable percentile calc

DEFAULT_FRAME_LENGTH = 1024  # frame size for analysis
DEFAULT_HOP_LENGTH = 256  # step size between frames

# Frame-level silence threshold for per-frame RMS checks.
# analyze_energy now receives the ORIGINAL (unnormalized) signal,
# so this threshold must work for raw audio levels from real devices.
FRAME_SILENCE_THRESHOLD = 0.005


# ---------------- GLOBAL ANALYSIS ----------------
def analyze_energy(
    y: np.ndarray,  # audio signal (already normalized by caller)
    sr: int,  # sample rate
    frame_length: int = DEFAULT_FRAME_LENGTH,  # frame size
    hop_length: int = DEFAULT_HOP_LENGTH,  # step size
) -> Dict:

    # ---------- PRE-ANALYSIS RMS GUARD ----------
    # whisper_service normalizes y to RMS ~0.05 before calling us.
    # Real speech after normalization sits at ~0.05; background noise that
    # slipped through will be amplified to a similar level, so we re-check
    # the raw signal RMS here as a secondary gate before any scoring.
    raw_rms = float(np.sqrt(np.mean(y ** 2)))
    if raw_rms < FRAME_SILENCE_THRESHOLD:
        return {
            "average_volume_db": -80.0,
            "dynamic_range": 0.0,
            "loudness_status": "Silence",
            "is_low_variation": True,
            "is_monotone": True,
        }

    # ---------- FRAME-LEVEL RMS ----------
    rms_energy = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    ).squeeze()  # remove extra dimensions

    # ratio of frames above silence threshold
    active_ratio = float(np.mean(rms_energy > FRAME_SILENCE_THRESHOLD))

    # if mostly silence across frames
    if active_ratio < MIN_ACTIVE_RATIO:
        return {
            "average_volume_db": -80.0,
            "dynamic_range": 0.0,
            "loudness_status": "Silence",
            "is_low_variation": True,
            "is_monotone": True,
        }

    # ---------- LOUDNESS ----------
    avg_power = np.mean(rms_energy ** 2)
    average_db = float(10 * np.log10(max(avg_power, 1e-10)))

    if average_db < WHISPER_THRESHOLD_DB:
        loudness_status = "Too quiet (whispering)"
    elif average_db > SHOUT_THRESHOLD_DB:
        loudness_status = "Too loud (shouting)"
    else:
        loudness_status = "Normal volume"

    # ---------- DYNAMIC RANGE ----------
    rms_energy = np.clip(rms_energy, 1e-10, None)
    db_levels = librosa.amplitude_to_db(rms_energy, ref=1.0)

    if len(db_levels) >= 2:
        if len(db_levels) >= MIN_FRAMES_FOR_ROBUST_PERCENTILE:
            low_percentile, high_percentile = 5, 95
        else:
            low_percentile, high_percentile = 0, 100

        dynamic_range = float(
            np.percentile(db_levels, high_percentile)
            - np.percentile(db_levels, low_percentile)
        )
    else:
        dynamic_range = 0.0

    is_low_variation = dynamic_range < LOW_VARIATION_DB

    # ---------- PITCH ----------
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"),
        )

        voiced_f0 = f0[voiced_flag]
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        is_monotone = pitch_std < MONOTONE_PITCH_STD_THRESHOLD

    except Exception:
        pitch_std = 0.0
        is_monotone = True

    return {
        "average_volume_db": round(average_db, 2),
        "dynamic_range": round(dynamic_range, 2),
        "loudness_status": loudness_status,
        "is_low_variation": is_low_variation,
        "pitch_variation_hz": round(pitch_std, 2),
        "is_monotone": is_monotone,
    }


# ---------------- PER-WORD ANALYSIS ----------------

def extract_audio_segment(y, sr, start, end):
    start_sample = int(start * sr)  # convert start time to sample index
    end_sample = int(end * sr)  # convert end time to sample index
    return y[start_sample:end_sample]  # slice audio segment


def compute_segment_db(y_segment):
    if len(y_segment) == 0:
        return -80.0  # empty segment → very low dB

    segment_rms = librosa.feature.rms(y=y_segment).squeeze()

    if len(segment_rms) == 0:
        return -80.0

    avg_power = np.mean(segment_rms ** 2)
    db = 10 * np.log10(max(avg_power, 1e-10))
    return float(db)


def classify_loudness(db):
    if db < WHISPER_THRESHOLD_DB:
        return "whisper"
    elif db > SHOUT_THRESHOLD_DB:
        return "shout"
    else:
        return "normal"


def analyze_per_word(y, sr, whisper_result) -> List[Dict]:
    word_results = []

    for segment in whisper_result.get("segments", []):
        for word_info in segment.get("words", []):
            start = word_info["start"]
            end = word_info["end"]
            word = word_info["word"]

            y_segment = extract_audio_segment(y, sr, start, end)
            db = compute_segment_db(y_segment)
            label = classify_loudness(db)

            word_results.append({
                "word": word,
                "start": round(start, 2),
                "end": round(end, 2),
                "volume_db": round(db, 2),
                "loudness": label,
            })

    return word_results