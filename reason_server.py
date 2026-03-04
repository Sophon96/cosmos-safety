"""
Cosmos Reason VLM server for remote inference.

Run on cloud: uvicorn reason_server:app --host 0.0.0.0 --port 8000

Local client sets COSMOS_REMOTE_URL=http://<cloud>:8000 to use this server.

Test with curl (replace localhost:8000 and video.mp4 as needed):

  # Health check
  curl http://localhost:8000/health

  # Binary check (is robot about to pour?)
  curl -X POST http://localhost:8000/binary_check -F "video=@videos/temp/episode_000_clip_13.0-14.0_30fps_1.0s.mp4" -F "fps=4"

  # Full reason (trajectory validation)
  curl -X POST http://localhost:8000/full_reason -F "video=@videos/temp/episode_000_clip_13.0-14.0_30fps_1.0s.mp4" -F "fps=4" -F "max_new_tokens=4096"

  # Full reason with custom prompt
  curl -X POST http://localhost:8000/full_reason -F "video=@video.mp4" -F "fps=4" -F "prompt=Describe what happens in this video." -F "max_new_tokens=4096"

  # List saved videos
  curl http://localhost:8000/videos

  # Download a saved video (use filename from /videos)
  curl -O -J http://localhost:8000/videos/FILENAME.mp4
"""

import logging
import os
import tempfile
import time
import uuid
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Unset COSMOS_REMOTE_URL so the server always uses local inference
os.environ.pop("COSMOS_REMOTE_URL", None)

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from reason import cosmos_binary_check, cosmos_full_reason, load_cosmos_model

app = FastAPI(title="Cosmos Reason VLM", version="0.1.0")

# Pre-load model at startup
_model, _processor = load_cosmos_model()

SAVED_VIDEOS_DIR = Path("saved_videos")
SAVED_VIDEOS_DIR.mkdir(exist_ok=True)


def _save_video(content: bytes, endpoint: str) -> Path:
    """Save video to disk and return the path."""
    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{endpoint}.mp4"
    path = SAVED_VIDEOS_DIR / filename
    path.write_bytes(content)
    return path


@app.post("/binary_check")
async def binary_check(
    video: UploadFile = File(...),
    fps: int = Form(default=4),
) -> dict:
    """
    Run rapid 1-token inference: is robot about to pour?
    Returns {"result": 0|1} where 1 = about to pour.
    """
    content = await video.read()
    _save_video(content, "binary_check")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    try:
        result = cosmos_binary_check(path, model=_model, processor=_processor, fps=fps)
        return {"result": result}
    finally:
        path.unlink(missing_ok=True)


@app.post("/full_reason")
async def full_reason(
    video: UploadFile = File(...),
    fps: int = Form(default=4),
    prompt: str | None = Form(default=None),
    max_new_tokens: int = Form(default=10240),
) -> dict:
    """
    Run full reasoning on video for trajectory validation.
    Returns {"output": "..."}. If prompt is omitted, uses prompt.txt.
    """
    content = await video.read()
    _save_video(content, "full_reason")
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(content)
        path = Path(f.name)
    try:
        if prompt is not None:
            # Use provided prompt text; need to pass via a temp file for reason.py API
            with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as pf:
                pf.write(prompt)
                prompt_path = Path(pf.name)
            try:
                output = cosmos_full_reason(
                    path,
                    prompt_path=prompt_path,
                    model=_model,
                    processor=_processor,
                    fps=fps,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                prompt_path.unlink(missing_ok=True)
        else:
            output = cosmos_full_reason(
                path,
                prompt_path="prompt.txt",
                model=_model,
                processor=_processor,
                fps=fps,
                max_new_tokens=max_new_tokens,
            )
        return {"output": output}
    finally:
        path.unlink(missing_ok=True)


@app.get("/videos")
async def list_videos() -> dict:
    """
    List all videos saved from inference requests.
    Returns {"videos": [{"filename": str, "endpoint": str, "timestamp": int}, ...]}.
    """
    videos = []
    for path in sorted(SAVED_VIDEOS_DIR.glob("*.mp4"), reverse=True):
        # Format: {timestamp}_{uuid}_{endpoint}.mp4 (endpoint may contain underscores)
        parts = path.stem.split("_")
        if len(parts) >= 3:
            ts_str = parts[0]
            endpoint = "_".join(parts[2:])
            try:
                timestamp = int(ts_str)
            except ValueError:
                timestamp = int(path.stat().st_mtime)
        else:
            endpoint = "unknown"
            timestamp = int(path.stat().st_mtime)
        videos.append({
            "filename": path.name,
            "endpoint": endpoint,
            "timestamp": timestamp,
        })
    return {"videos": videos}


@app.get("/videos/{filename}")
async def get_video(filename: str):
    """
    Download a specific saved video by filename.
    Use GET /videos to get the list of filenames.
    """
    path = SAVED_VIDEOS_DIR / filename
    if not path.exists() or not path.is_file():
        raise HTTPException(status_code=404, detail="Video not found")
    return FileResponse(path, media_type="video/mp4", filename=filename)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
