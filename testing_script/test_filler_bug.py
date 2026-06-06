import sys
import json

# Setup path
sys.path.append('d:/iSpeak_Thesis_Project/ispeak_python-backend')

from audio_analysis_processing_files.clarity_analysis_module.voice_fillerwords_detection import analyze_fillers

# Dummy Whisper segments containing fillers
segments = [
    {"text": "um", "start": 0.0, "end": 0.5, "confidence": 1.0},
    {"text": "I", "start": 0.5, "end": 0.8, "confidence": 1.0},
    {"text": "think", "start": 0.8, "end": 1.2, "confidence": 1.0},
    {"text": "like", "start": 1.2, "end": 1.5, "confidence": 1.0},
    {"text": "it", "start": 1.5, "end": 1.8, "confidence": 1.0},
    {"text": "is", "start": 1.8, "end": 2.0, "confidence": 1.0},
    {"text": "like", "start": 2.0, "end": 2.3, "confidence": 1.0},
    {"text": "you know", "start": 2.3, "end": 2.8, "confidence": 1.0},
    {"text": "good", "start": 2.8, "end": 3.0, "confidence": 1.0},
]

res = analyze_fillers(segments)
print(json.dumps(res, indent=2))
