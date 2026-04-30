# model.py
"""import whisper

def load_model(model_name: str = "base"):
    return whisper.load_model(model_name)"""
    

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
            provider="CUDAExecutionProvider" # This forces the GPU to take over!
        )
        
        # The pipeline automatically handles audio processing and word-level timestamps
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            return_timestamps="word"
        )
        print("Custom model loaded and ready!")

    def transcribe(self, file_path, **kwargs):
        """
        Runs the ONNX model and reformats the output to perfectly match 
        the standard OpenAI Whisper format so whisper_service.py doesn't break.
        """
        output = self.pipe(file_path)
        
        formatted_words = []
        for chunk in output.get("chunks", []):
            # Extract start and end times from the HF 'timestamp' tuple
            start, end = chunk.get("timestamp", (0.0, 0.0))
            
            # Hugging Face sometimes leaves the very last timestamp open (None)
            if end is None:
                end = start + 0.5
                
            formatted_words.append({
                "word": chunk.get("text"),
                "start": start,
                "end": end,
                "probability": 1.0  # Default confidence score
            })
            
        return {
            "text": output.get("text", ""),
            # Wrap in the 'segments'/'words' structure that _extract_word_segments expects
            "segments": [{"words": formatted_words}] 
        }

def load_model(model_name: str = "models/iSpeak_v3/model_files"):
    """
    By setting the default model_name to your new folder path, 
    fastapi_backend.py will load your custom model automatically!
    """
    return OptimizedONNXWhisper(model_name)
