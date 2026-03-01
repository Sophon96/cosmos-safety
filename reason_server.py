"""
Cosmos Reason VLM server for remote inference.

Run on cloud: uvicorn reason_server:app --host 0.0.0.0 --port 8000

Local client sets COSMOS_REMOTE_URL=http://<cloud>:8000 to use this server.
"""

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile

from reason import cosmos_binary_check, cosmos_full_reason

app = FastAPI(title="Cosmos Reason VLM", version="0.1.0")


@app.post("/binary_check")
async def binary_check(
    video: UploadFile = File(...),
    fps: int = Form(default=4),
) -> dict:
    """
    Run rapid 1-token inference: is robot about to pour?
    Returns {"result": 0|1} where 1 = about to pour.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        content = await video.read()
        f.write(content)
        path = Path(f.name)
    try:
        result = cosmos_binary_check(path, fps=fps)
        return {"result": result}
    finally:
        path.unlink(missing_ok=True)


@app.post("/full_reason")
async def full_reason(
    video: UploadFile = File(...),
    fps: int = Form(default=4),
    prompt: str | None = Form(default=None),
    max_new_tokens: int = Form(default=512),
) -> dict:
    """
    Run full reasoning on video for trajectory validation.
    Returns {"output": "..."}. If prompt is omitted, uses prompt.txt.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        content = await video.read()
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
                    fps=fps,
                    max_new_tokens=max_new_tokens,
                )
            finally:
                prompt_path.unlink(missing_ok=True)
        else:
            output = cosmos_full_reason(
                path,
                prompt_path="prompt.txt",
                fps=fps,
                max_new_tokens=max_new_tokens,
            )
        return {"output": output}
    finally:
        path.unlink(missing_ok=True)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok"}
