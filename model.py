# model.py
import os
import librosa
from transformers import pipeline, WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

class OptimizedONNXWhisper:
    def __init__(self, model_path: str):
        print(f"Loading custom ONNX model from {model_path} into RTX 4050...")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            provider="CUDAExecutionProvider", # This forces the GPU to take over!
            use_merged=False
        )
        
        # Give the dummy wrapper a copy of the config so the pipeline doesn't crash
        self.model.model.config = self.model.config
        
        # The pipeline automatically handles audio processing
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            return_timestamps=True # CHANGED: Phrase-level timestamps bypass the cross-attention crash!
        )
        print("Custom model loaded and ready!")

    def transcribe(self, file_path, **kwargs):
        output = self.pipe(file_path)
        
        formatted_words = []
        full_text = output.get("text", "").strip()
        
        for chunk in output.get("chunks", []):
            start, end = chunk.get("timestamp", (0.0, 0.0))
            if end is None: end = start + 1.0
                
            phrase = chunk.get("text", "").strip()
            words = phrase.split()
            if not words: continue
                
            word_duration = (end - start) / len(words)
            current_start = start
            
            for w in words:
                formatted_words.append({
                    "word": w,
                    "start": round(current_start, 2),
                    "end": round(current_start + word_duration, 2),
                    "probability": 1.0 
                })
                current_start += word_duration
        
        # Determine total boundaries for the segment
        total_start = formatted_words[0]["start"] if formatted_words else 0.0
        total_end = formatted_words[-1]["end"] if formatted_words else 0.0

        return {
            "text": full_text,
            "segments": [{
                "text": full_text,
                "start": total_start,
                "end": total_end,
                "words": formatted_words
            }],
            "duration": total_end # This helps whisper_service skip the librosa fallback!
        }

def load_model(model_name: str = "models/iSpeak_v3/model_files"):
    return OptimizedONNXWhisper(model_name)