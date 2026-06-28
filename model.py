# model.py
import os
import re
import librosa
import onnxruntime as ort
from collections import Counter
from transformers import pipeline, WhisperProcessor
from optimum.onnxruntime import ORTModelForSpeechSeq2Seq

class OptimizedONNXWhisper:
    def __init__(self, model_path: str):
        # Pick the best available execution provider
        available = ort.get_available_providers()
        if "CUDAExecutionProvider" in available:
            provider = "CUDAExecutionProvider"
            device_label = "GPU (CUDA)"
        else:
            provider = "CPUExecutionProvider"
            device_label = "CPU"

        print(f"Loading custom ONNX model from {model_path} on {device_label}...")
        self.processor = WhisperProcessor.from_pretrained(model_path)
        self.model = ORTModelForSpeechSeq2Seq.from_pretrained(
            model_path, 
            provider=provider,
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
        # Build generate_kwargs to suppress hallucinations
        generate_kwargs = {
            "no_repeat_ngram_size": 3,           # Prevents repeating 3-gram phrases
            "repetition_penalty": 1.2,           # Penalizes repeated tokens
            "language": "en",                    # Explicit language avoids detection overhead
            "task": "transcribe",                # Explicit task avoids translation
        }

        output = self.pipe(
            file_path,
            generate_kwargs=generate_kwargs,
        )

        formatted_words = []
        full_text = output.get("text", "").strip()

        for chunk in output.get("chunks", []):
            start, end = chunk.get("timestamp", (0.0, 0.0))
            if end is None:
                end = start + 1.0

            phrase = chunk.get("text", "").strip()
            words = phrase.split()
            if not words:
                continue

            word_duration = (end - start) / len(words)
            current_start = start

            for w in words:
                formatted_words.append({
                    "word": w,
                    "start": round(current_start, 2),
                    "end": round(current_start + word_duration, 2),
                    "probability": 0.9  # Phrase-level timestamps don't give per-word probs
                })
                current_start += word_duration

        # --- Post-transcription hallucination / repetition-loop detector ---
        formatted_words, full_text = self._trim_repetition_loop(formatted_words)

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
            "duration": total_end
        }

    @staticmethod
    def _trim_repetition_loop(
        words: list,
        max_ngram: int = 6,
        min_repeats: int = 3,
    ) -> tuple:
        """
        Detect and remove repetition loops from the word list.

        Scans for repeated n-gram phrases (1..max_ngram words). If the same
        n-gram appears >= min_repeats times consecutively, only the first
        occurrence is kept and the rest are discarded.

        Returns (trimmed_words, trimmed_text).
        """
        if len(words) < min_repeats * 2:
            text = " ".join(w["word"] for w in words)
            return words, text

        word_strings = [w["word"].strip().lower() for w in words]
        best_cut = len(words)  # default: no cut

        for ngram_len in range(1, max_ngram + 1):
            for start_idx in range(len(word_strings) - ngram_len * min_repeats + 1):
                pattern = tuple(word_strings[start_idx : start_idx + ngram_len])
                repeat_count = 0
                pos = start_idx
                while pos + ngram_len <= len(word_strings):
                    window = tuple(word_strings[pos : pos + ngram_len])
                    if window == pattern:
                        repeat_count += 1
                        pos += ngram_len
                    else:
                        break

                if repeat_count >= min_repeats:
                    # Keep only the first occurrence of the repeated pattern
                    cut_at = start_idx + ngram_len
                    if cut_at < best_cut:
                        best_cut = cut_at

        if best_cut < len(words):
            print(f"[hallucination-guard] Repetition loop detected — trimming from "
                  f"{len(words)} words to {best_cut} words")
            words = words[:best_cut]

        text = " ".join(w["word"] for w in words)
        return words, text

def load_model(model_name: str = "models/iSpeak_v3/model_files"):
    return OptimizedONNXWhisper(model_name)