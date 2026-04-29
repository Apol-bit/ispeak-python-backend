import os
import shutil
import tempfile
import logging
import traceback
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.concurrency import run_in_threadpool

from whisper_service import generate_full_analysis
from model import load_model

# Configuration

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# App lifespan (startup/shutdown)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("App started")
    app.state.model = load_model("base")
    yield


app = FastAPI(lifespan=lifespan)

# Endpoint

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):

    suffix = os.path.splitext(file.filename)[1].lower()

    if not suffix:
        raise HTTPException(status_code=400, detail="File must have an extension")

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp:
            temp_path = temp.name
            shutil.copyfileobj(file.file, temp)

        result = await run_in_threadpool(
            generate_full_analysis,
            temp_path,
            app.state.model  
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
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError as e:
                logger.warning(
                    "Failed to remove temp file %s: %s",
                    temp_path,
                    e
                )