# voice_energy_analyze.py
import numpy as np  # numerical operations
import librosa  # audio processing library
from typing import Dict, List  # type hints


# ---------------- CONFIG ----------------

WHISPER_THRESHOLD_DB = -30.0  # below this = whispering
SHOUT_THRESHOLD_DB = -7.0  # above this = shouting

LOW_VARIATION_DB = 6.0  # how much loudness changes over time. variation < 12 dB → speech is flat / dull. talks like robot.
MONOTONE_PITCH_STD_THRESHOLD = 20.0  # how much pitch (voice tone) changes. if < 20 Hz → monotone (boring, flat, no emotion).

MIN_ACTIVE_RATIO = 0.10  # 0.10 = at least 10% of audio must be non-silent. If below this: Audio is treated as mostly silence
MIN_FRAMES_FOR_ROBUST_PERCENTILE = 50  # min frames for stable percentile calc. how much data needed before trusting advanced calculations

DEFAULT_FRAME_LENGTH = 1024  # frame size for analysis. how big each frame size is
DEFAULT_HOP_LENGTH = 256  # step size between frames. how much each frame size move each step (creates overlap)


# ---------------- GLOBAL ANALYSIS ----------------
def analyze_energy(
    y: np.ndarray,  # audio signal
    sr: int,  # sample rate
    silence_threshold: float = 1e-4,  # threshold to detect silence
    frame_length: int = DEFAULT_FRAME_LENGTH,  # frame size
    hop_length: int = DEFAULT_HOP_LENGTH,  # step size
) -> Dict:

    # compute RMS energy per frame
    rms_energy = librosa.feature.rms(
        y=y,
        frame_length=frame_length,
        hop_length=hop_length,
    ).squeeze()  # remove extra dimensions

    # ratio of frames above silence threshold
    active_ratio = float(np.mean(rms_energy > silence_threshold))

    # if mostly silence
    if active_ratio < MIN_ACTIVE_RATIO:
        return {
            "average_volume_db": -80.0,  # very low volume
            "dynamic_range": 0.0,  # no variation
            "loudness_status": "Silence",  # label
            "is_low_variation": True,  # flat signal
            "is_monotone": True,  # no pitch variation
        }

    # compute average power
    avg_power = np.mean(rms_energy ** 2)

    # convert to decibels
    average_db = float(10 * np.log10(max(avg_power, 1e-10)))

    # classify loudness
    if average_db < WHISPER_THRESHOLD_DB:
        loudness_status = "Too quiet (whispering)"

    elif average_db > SHOUT_THRESHOLD_DB:
        loudness_status = "Too loud (shouting)"

    else:
        loudness_status = "Normal volume"

    # avoid log(0)
    rms_energy = np.clip(rms_energy, 1e-10, None)

    # convert amplitude to dB scale
    db_levels = librosa.amplitude_to_db(rms_energy, ref=1.0)

    # compute dynamic range if enough data
    if len(db_levels) >= 2:
        if len(db_levels) >= MIN_FRAMES_FOR_ROBUST_PERCENTILE:
            low_percentile, high_percentile = 5, 95  # robust percentiles
        else:
            low_percentile, high_percentile = 0, 100  # full range

        # dynamic range = high percentile - low percentile
        dynamic_range = float(
            np.percentile(db_levels, high_percentile)
            - np.percentile(db_levels, low_percentile)
        )
    else:
        dynamic_range = 0.0  # not enough data

    # check if variation is low
    is_low_variation = dynamic_range < LOW_VARIATION_DB

    try:
        # estimate pitch using pyin
        f0, voiced_flag, _ = librosa.pyin(
            y,
            fmin=librosa.note_to_hz("C2"),  # low pitch bound
            fmax=librosa.note_to_hz("C7"),  # high pitch bound
        )

        # keep only voiced frames
        voiced_f0 = f0[voiced_flag]

        # compute pitch variation (std deviation)
        pitch_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        # monotone if low variation
        is_monotone = pitch_std < MONOTONE_PITCH_STD_THRESHOLD

    except Exception:
        pitch_std = 0.0  # fallback if error
        is_monotone = True  # assume monotone

    # return results
    return {
        "average_volume_db": round(average_db, 2),  # avg loudness
        "dynamic_range": round(dynamic_range, 2),  # variation
        "loudness_status": loudness_status,  # label
        "is_low_variation": is_low_variation,  # low variation flag
        "pitch_variation_hz": round(pitch_std, 2),  # pitch std
        "is_monotone": is_monotone,  # monotone flag
    }


# ---------------- PER-WORD ANALYSIS ----------------

def extract_audio_segment(y, sr, start, end):
    start_sample = int(start * sr)  # convert start time to sample index
    end_sample = int(end * sr)  # convert end time to sample index
    return y[start_sample:end_sample]  # slice audio segment


def compute_segment_db(y_segment):
    if len(y_segment) == 0:
        return -80.0  # empty segment → very low dB

    # compute root-mean-square (RMS) for segment
    segment_rms = librosa.feature.rms(y=y_segment).squeeze()

    if len(segment_rms) == 0:
        return -80.0  # no frames → low dB

    # compute average power
    avg_power = np.mean(segment_rms ** 2)

    # convert to dB
    db = 10 * np.log10(max(avg_power, 1e-10))
    return float(db)


def classify_loudness(db):
    if db < WHISPER_THRESHOLD_DB:
        return "whisper"  # too quiet
    elif db > SHOUT_THRESHOLD_DB:
        return "shout"  # too loud
    else:
        return "normal"  # acceptable


def analyze_per_word(y, sr, whisper_result) -> List[Dict]:
    word_results = []  # store results

    # loop through segments from whisper
    for segment in whisper_result.get("segments", []):
        # loop through words in each segment
        for word_info in segment.get("words", []):
            start = word_info["start"]  # word start time
            end = word_info["end"]  # word end time
            word = word_info["word"]  # word text

            # extract audio for this word
            y_segment = extract_audio_segment(y, sr, start, end)

            # compute loudness
            db = compute_segment_db(y_segment)

            # classify loudness
            label = classify_loudness(db)

            # store result
            word_results.append({
                "word": word,
                "start": round(start, 2),  # rounded start time
                "end": round(end, 2),  # rounded end time
                "volume_db": round(db, 2),  # loudness in dB
                "loudness": label  # label
            })

    return word_results  # return all word data