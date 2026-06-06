"""
iSpeak Offline AI Evaluation Tool
===================================
Run the full speech analysis pipeline locally without the server.

Usage:
    # Single file
    python offline_evaluation.py audio.wav

    # Folder of audio files
    python offline_evaluation.py ./my_audio_folder

    # Save JSON report
    python offline_evaluation.py audio.wav --output results.json

    # Quick mode (skip heavy model load, only run librosa modules)
    python offline_evaluation.py audio.wav --quick
"""

from __future__ import annotations

import io
import argparse
import json
import os
import sys
import time
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import librosa

# ---------------------------------------------------------------------------
# Make sure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from whisper_service import (
    generate_full_analysis,
    _compute_energy_score,
    _compute_pacing_score,
    _compute_overall_score,
    _extract_word_segments,
    RMS_SILENCE_THRESHOLD,
    MIN_WORD_COUNT,
)
from audio_analysis_processing_files.voice_energy_analyze import analyze_energy
from audio_analysis_processing_files.voice_pacing_calculation import calculate_pacing
from audio_analysis_processing_files.clarity_analysis_module.voice_clarity_detection import (
    analyze_pronunciation,
)
from audio_analysis_processing_files.clarity_analysis_module.voice_fillerwords_detection import (
    analyze_fillers,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

# Force UTF-8 stdout on Windows to avoid cp1252 UnicodeEncodeError
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("offline_eval")

# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------

RESET   = "\033[0m"
BOLD    = "\033[1m"
GREEN   = "\033[92m"
YELLOW  = "\033[93m"
RED     = "\033[91m"
CYAN    = "\033[96m"
MAGENTA = "\033[95m"
DIM     = "\033[2m"


def _score_color(score: float) -> str:
    """Pick a color based on the score value."""
    if score >= 85:
        return GREEN
    elif score >= 60:
        return YELLOW
    else:
        return RED


def _bar(score: float, width: int = 30) -> str:
    """Build a visual progress bar string."""
    filled = int(round(score / 100 * width))
    color = _score_color(score)
    return f"{color}{'#' * filled}{DIM}{'-' * (width - filled)}{RESET}"


def _print_separator(char: str = "-", width: int = 60):
    print(f"{DIM}{char * width}{RESET}")


def _print_header(text: str):
    print()
    _print_separator("=")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    _print_separator("=")


def _print_section(title: str):
    print()
    print(f"  {BOLD}{MAGENTA}> {title}{RESET}")
    _print_separator("-", 50)


def _print_score_line(label: str, score: float, extra: str = ""):
    color = _score_color(score)
    bar = _bar(score)
    extra_str = f"  {DIM}{extra}{RESET}" if extra else ""
    print(f"    {label:<22} {bar} {color}{score:>6.1f}{RESET}{extra_str}")


# ---------------------------------------------------------------------------
# Quick (no-model) analysis — only librosa-based modules
# ---------------------------------------------------------------------------

def _quick_evaluate(file_path: str) -> Dict[str, Any]:
    """Run energy analysis only (no model required)."""
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)

    rms = float(np.sqrt(np.mean(y ** 2)))
    if rms < RMS_SILENCE_THRESHOLD:
        return {
            "file": file_path,
            "duration_seconds": round(duration, 2),
            "rms": round(rms, 6),
            "status": "silence",
            "message": "Audio below silence threshold -- no speech detected.",
        }

    energy_stats = analyze_energy(y, sr)
    energy_score = _compute_energy_score(energy_stats)

    return {
        "file": file_path,
        "duration_seconds": round(duration, 2),
        "rms": round(rms, 6),
        "status": "analyzed",
        "energy_stats": energy_stats,
        "energy_score": energy_score,
        "note": "Quick mode -- only energy (librosa) analysis. Use full mode for transcription + scoring.",
    }


# ---------------------------------------------------------------------------
# Full pipeline evaluation
# ---------------------------------------------------------------------------

def _full_evaluate(file_path: str, model) -> Dict[str, Any]:
    """Run the complete iSpeak pipeline on a single audio file."""
    t0 = time.perf_counter()
    result = generate_full_analysis(file_path, model)
    elapsed = time.perf_counter() - t0

    # Enrich with metadata
    y, sr = librosa.load(file_path, sr=16000, mono=True)
    duration = librosa.get_duration(y=y, sr=sr)
    rms = float(np.sqrt(np.mean(y ** 2)))

    return {
        "file": os.path.basename(file_path),
        "path": file_path,
        "duration_seconds": round(duration, 2),
        "rms": round(rms, 6),
        "processing_time_seconds": round(elapsed, 2),
        **result,
    }


# ---------------------------------------------------------------------------
# Console report
# ---------------------------------------------------------------------------

def _print_report(result: Dict[str, Any]):
    """Pretty-print the evaluation result to the console."""
    _print_header(f"[FILE] {result.get('file', result.get('path', 'Unknown'))}")

    # Metadata
    print(f"    Duration   : {result['duration_seconds']}s")
    print(f"    RMS Level  : {result['rms']}")
    if "processing_time_seconds" in result:
        print(f"    Processed  : {result['processing_time_seconds']}s")

    # Quick mode
    if result.get("status") == "silence":
        print(f"\n    {RED}[!] {result['message']}{RESET}")
        return
    if "note" in result:
        _print_section("Energy (Quick Mode)")
        energy = result.get("energy_stats", {})
        _print_score_line("Energy Score", result.get("energy_score", 0))
        print(f"      Volume   : {energy.get('average_volume_db', 'N/A')} dB")
        print(f"      Loudness : {energy.get('loudness_status', 'N/A')}")
        print(f"      Range    : {energy.get('dynamic_range', 'N/A')} dB")
        print(f"      Monotone : {'Yes' if energy.get('is_monotone') else 'No'}")
        print(f"\n    {DIM}{result['note']}{RESET}")
        return

    # Full mode — Transcription
    text = result.get("transcription", "")
    if text:
        _print_section("Transcription")
        # Wrap long text
        words = text.split()
        lines = []
        current = "      "
        for w in words:
            if len(current) + len(w) + 1 > 75:
                lines.append(current)
                current = "      " + w
            else:
                current += " " + w if current.strip() else w
        lines.append(current)
        for line in lines:
            print(line)

    # Scores overview
    scores = result.get("scores", {})
    if scores:
        _print_section("Scores")
        _print_score_line("Overall",  scores.get("overall", 0))
        _print_score_line("Clarity",  scores.get("clarity", 0))
        _print_score_line("Pacing",   scores.get("pacing", 0))
        _print_score_line("Energy",   scores.get("energy", 0))

    # Pacing details
    pacing = result.get("pacing", {})
    if pacing:
        _print_section("Pacing Details")
        wpm = pacing.get("wpm", 0)
        msg = pacing.get("message", "")
        print(f"      WPM      : {wpm}")
        print(f"      Status   : {msg}")

    # Pronunciation details
    pron = result.get("pronunciation", {})
    if pron:
        _print_section("Pronunciation")
        print(f"      Score    : {pron.get('score', 0)}")
        print(f"      Feedback : {pron.get('message', '')}")
        problems = pron.get("problematic_words", [])
        if problems:
            print(f"      Issues   : {len(problems)} word(s)")
            for pw in problems[:10]:  # show top 10
                print(f"        - {pw['word']:<15} conf={pw['confidence']:.2f}  dur={pw['duration']:.2f}s  ({pw['issue']})")
            if len(problems) > 10:
                print(f"        ... and {len(problems) - 10} more")

    # Filler details
    fillers = result.get("fillers", {})
    if fillers:
        _print_section("Filler Words")
        print(f"      Score    : {fillers.get('score', 0)}")
        print(f"      Count    : {fillers.get('count', 0)}")
        print(f"      Rate     : {fillers.get('rate', 0.0)}%")
        print(f"      Feedback : {fillers.get('message', '')}")
        filler_words = fillers.get("words", [])
        if filler_words:
            for fw in filler_words[:10]:
                start = fw.get('start', 0.0)
                end = fw.get('end', 0.0)
                print(f"        - \"{fw['word']}\"  ({start:.2f}s – {end:.2f}s)")

    print()


# ---------------------------------------------------------------------------
# Batch summary
# ---------------------------------------------------------------------------

def _print_batch_summary(results: List[Dict[str, Any]]):
    """Print a compact summary table for batch evaluation."""
    full_results = [r for r in results if "scores" in r]
    if not full_results:
        return

    _print_header("[SUMMARY] Batch Results")

    # Header
    print(f"    {'File':<30} {'Overall':>8} {'Clarity':>8} {'Pacing':>8} {'Energy':>8} {'WPM':>6}")
    _print_separator("-", 80)

    for r in full_results:
        scores = r.get("scores", {})
        wpm = r.get("pacing", {}).get("wpm", 0)
        name = r.get("file", "?")
        if len(name) > 28:
            name = name[:25] + "..."

        overall_color = _score_color(scores.get("overall", 0))
        print(
            f"    {name:<30}"
            f" {overall_color}{scores.get('overall', 0):>7.1f}{RESET}"
            f" {scores.get('clarity', 0):>7.1f}"
            f" {scores.get('pacing', 0):>7.1f}"
            f" {scores.get('energy', 0):>7.1f}"
            f" {wpm:>6.1f}"
        )

    # Averages
    if len(full_results) > 1:
        _print_separator("-", 80)
        avg = lambda key: sum(r["scores"].get(key, 0) for r in full_results) / len(full_results)
        avg_wpm = sum(r.get("pacing", {}).get("wpm", 0) for r in full_results) / len(full_results)
        avg_overall = avg("overall")
        overall_color = _score_color(avg_overall)
        print(
            f"    {'AVERAGE':<30}"
            f" {overall_color}{avg_overall:>7.1f}{RESET}"
            f" {avg('clarity'):>7.1f}"
            f" {avg('pacing'):>7.1f}"
            f" {avg('energy'):>7.1f}"
            f" {avg_wpm:>6.1f}"
        )

    print()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def _discover_audio_files(path: str) -> List[str]:
    """Return a list of audio file paths from a file or directory."""
    p = Path(path)
    if p.is_file():
        if p.suffix.lower() in SUPPORTED_EXTENSIONS:
            return [str(p)]
        else:
            logger.error("Unsupported file type: %s (supported: %s)", p.suffix, SUPPORTED_EXTENSIONS)
            return []

    if p.is_dir():
        files = sorted(
            str(f) for f in p.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        if not files:
            logger.error("No audio files found in %s", path)
        return files

    logger.error("Path does not exist: %s", path)
    return []


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="iSpeak Offline AI Evaluation — run the full speech analysis pipeline locally.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "input",
        help="Audio file or directory of audio files to evaluate.",
    )
    parser.add_argument(
        "--output", "-o",
        help="Path to save results as JSON (optional).",
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Quick mode: skip model loading, only run energy (librosa) analysis.",
    )
    parser.add_argument(
        "--model",
        default="models/iSpeak_v3/model_files",
        help="Path to the ONNX model directory (default: models/iSpeak_v3/model_files).",
    )
    args = parser.parse_args()

    # Discover files
    audio_files = _discover_audio_files(args.input)
    if not audio_files:
        sys.exit(1)

    logger.info("Found %d audio file(s) to evaluate.", len(audio_files))

    # Load model (unless quick mode)
    model = None
    if not args.quick:
        logger.info("Loading ONNX model from %s ...", args.model)
        t0 = time.perf_counter()
        from model import load_model
        model = load_model(args.model)
        logger.info("Model loaded in %.1fs", time.perf_counter() - t0)

    # Process each file
    results: List[Dict[str, Any]] = []

    for i, fpath in enumerate(audio_files, 1):
        logger.info("[%d/%d] Processing: %s", i, len(audio_files), os.path.basename(fpath))
        try:
            if args.quick:
                result = _quick_evaluate(fpath)
            else:
                result = _full_evaluate(fpath, model)
            results.append(result)
            _print_report(result)
        except Exception as e:
            logger.error("Failed to process %s: %s", fpath, e, exc_info=True)
            results.append({"file": fpath, "error": str(e)})

    # Batch summary
    if len(results) > 1:
        _print_batch_summary(results)

    # Save JSON
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        logger.info("Results saved to %s", output_path)

    # Final summary line
    scored = [r for r in results if "scores" in r]
    if scored:
        avg_overall = sum(r["scores"]["overall"] for r in scored) / len(scored)
        color = _score_color(avg_overall)
        print(f"  {BOLD}Average Overall Score: {color}{avg_overall:.1f}/100{RESET}")
    
    errors = [r for r in results if "error" in r]
    if errors:
        print(f"  {RED}{len(errors)} file(s) failed to process.{RESET}")


if __name__ == "__main__":
    main()
