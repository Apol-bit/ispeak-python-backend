"""Quick test: pitch-based filler detection with synthetic audio."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from audio_analysis_processing_files.clarity_analysis_module.voice_fillerwords_detection import analyze_fillers

SR = 16000

def make_monotone(duration, freq=180, amp=0.03):
    """Simulate a filler sound: sustained monotone voiced (like 'um')."""
    t = np.linspace(0, duration, int(duration * SR), dtype=np.float32)
    return amp * np.sin(2 * np.pi * freq * t)

def make_speech(duration, amp=0.03):
    """Simulate speech: varying pitch (NOT monotone)."""
    t = np.linspace(0, duration, int(duration * SR), dtype=np.float32)
    # Pitch that varies from 150 to 300 Hz — realistic speech intonation
    freq = 150 + 150 * np.sin(2 * np.pi * 3 * t)  # 3 Hz modulation
    phase = np.cumsum(2 * np.pi * freq / SR)
    return (amp * np.sin(phase)).astype(np.float32)

def make_silence(duration):
    return np.zeros(int(duration * SR), dtype=np.float32)


def test(name, segments, y):
    r = analyze_fillers(segments, y=y, sr=SR)
    print(f"=== {name} ===")
    print(f"  Count: {r['filler_count']}, Rate: {r['filler_rate']}%, Score: {r['filler_score']}")
    for fw in r.get('filler_words', []):
        src = fw.get('source', '?')
        pstd = fw.get('_pitch_std', '?')
        print(f"    -> [{src}] \"{fw['word']}\" {fw['start']:.3f}s-{fw['end']:.3f}s (pitch_std={pstd})")
    print(f"  Message: {r['message']}\n")


# Test 1: Speech with a filler gap (monotone between speech)
# [speech 0-1s] [filler um 1-1.4s] [speech 1.4-2.4s]
y1 = np.concatenate([
    make_speech(1.0),
    make_monotone(0.4, freq=180),  # "um" — monotone
    make_speech(1.0),
])
segs1 = [
    {"text": "I",    "start": 0.0, "end": 0.5, "confidence": 1.0},
    {"text": "went", "start": 0.5, "end": 1.0, "confidence": 1.0},
    {"text": "to",   "start": 1.4, "end": 1.7, "confidence": 1.0},
    {"text": "school","start": 1.7, "end": 2.4, "confidence": 1.0},
]
test("Speech with monotone filler gap", segs1, y1)

# Test 2: All speech, no fillers
y2 = make_speech(3.0)
segs2 = [
    {"text": "The",     "start": 0.0, "end": 1.0, "confidence": 1.0},
    {"text": "weather", "start": 1.0, "end": 2.0, "confidence": 1.0},
    {"text": "today",   "start": 2.0, "end": 3.0, "confidence": 1.0},
]
test("All speech, no fillers", segs2, y2)

# Test 3: Multiple fillers
y3 = np.concatenate([
    make_speech(0.5),
    make_monotone(0.3, freq=170),
    make_speech(0.5),
    make_monotone(0.35, freq=190),
    make_speech(0.5),
])
segs3 = [
    {"text": "I",     "start": 0.0,  "end": 0.5,  "confidence": 1.0},
    {"text": "went",  "start": 0.8,  "end": 1.3,  "confidence": 1.0},
    {"text": "there", "start": 1.65, "end": 2.15, "confidence": 1.0},
]
test("Multiple monotone fillers", segs3, y3)

# Test 4: Empty
test("No speech", [], make_silence(1.0))

print("All tests done!")
