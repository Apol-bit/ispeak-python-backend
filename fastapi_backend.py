import os
import shutil
import tempfile
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.concurrency import run_in_threadpool

from whisper_service import generate_full_analysis, generate_reference_analysis
from model import load_model

# Configuration

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App lifespan (startup/shutdown)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App started")
    app.state.model = load_model()
    yield


app = FastAPI(lifespan=lifespan)


def _validate_extension(filename: str) -> str:
    """Validate and return the file extension."""
    suffix = os.path.splitext(filename)[1].lower()
    if not suffix:
        raise HTTPException(status_code=400, detail="File must have an extension")
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")
    return suffix


# Endpoint — supports optional reference audio for comparison scoring
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    reference_audio: UploadFile | None = File(default=None),
):
    suffix = _validate_extension(file.filename)

    temp_path = None
    ref_temp_path = None

    try:
        # Save user audio to temp file
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            shutil.copyfileobj(file.file, temp)

        # Save reference audio to temp file (if provided)
        if reference_audio is not None:
            ref_suffix = _validate_extension(reference_audio.filename)
            with tempfile.NamedTemporaryFile(suffix=ref_suffix, delete=False) as ref_temp:
                ref_temp_path = ref_temp.name
                shutil.copyfileobj(reference_audio.file, ref_temp)

        # Choose analysis method based on whether reference audio is provided
        if ref_temp_path:
            logger.info("Running reference-based analysis (user vs. validator audio)")
            result = await run_in_threadpool(
                generate_reference_analysis,
                temp_path,
                ref_temp_path,
                app.state.model,
            )
        else:
            logger.info("Running standard analysis (no reference audio)")
            result = await run_in_threadpool(
                generate_full_analysis,
                temp_path,
                app.state.model,
            )

        return result

    except Exception as e:
        logger.error(
            "Processing failed for %s:\n%s",
            file.filename,
            traceback.format_exc()
        )
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )

    finally:
        for p in (temp_path, ref_temp_path):
            if p and os.path.exists(p):
                try:
                    os.remove(p)
                except OSError as e:
                    logger.warning("Failed to remove temp file %s: %s", p, e)