# Unsloth must be imported before transformers for optimizations
# from unsloth import FastVisionModel
import os

# Use /workspace for HF cache on cloud GPUs where root disk is small
if not os.environ.get("HF_HOME") and os.path.exists("/workspace"):
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"

import logging
import torch
import transformers
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

MODEL_NAME ="nvidia/Cosmos-Reason2-8B"
BINARY_PROMPT = (
    "You are provided a video snippet. In the video there is a robot holding a red and white milk carton. It is trying to pour liquid into a brown cup"
    "Is the robot about to pour water from one cup into another, or in the process of pouring water?"
    "Reply with ONLY a single character: 1 if yes (about to pour), 0 if no."
)

_model_cache: tuple[Any, Any] | None = None


def load_cosmos_model(
    model_name: str = MODEL_NAME,
    device_map: str = "auto",
    dtype: torch.dtype | None = torch.float16,
) -> tuple[Any, Any]:
    """Load Cosmos model and processor. Uses singleton cache to avoid reloading."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache

    model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="sdpa",
    )
    processor = transformers.AutoProcessor.from_pretrained(model_name)
    _model_cache = (model, processor)
    return model, processor


def _frames_to_video_path(
    frames: list[Any],
    fps: int = 4,
    output_path: str | Path | None = None,
) -> Path:
    """Write frames to a temp mp4 file and return the path."""
    import os
    import tempfile

    import imageio

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".mp4")
        os.close(fd)
        output_path = Path(output_path)

    # frames: list of numpy arrays (H, W, 3) uint8
    writer = imageio.get_writer(str(output_path), fps=fps, codec="libx264")
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return Path(output_path)


def _binary_check_remote(url: str, video_path: str | Path, fps: int) -> int:
    """Call remote server for binary check. video_path must be a file path."""
    import httpx

    url = url.rstrip("/")
    with open(video_path, "rb") as f:
        files = {"video": (Path(video_path).name, f, "video/mp4")}
        data = {"fps": fps}
        with httpx.Client(timeout=30.0) as client:
            r = client.post(f"{url}/binary_check", files=files, data=data)
            r.raise_for_status()
            return int(r.json()["result"])


def cosmos_binary_check(
    video_path: str | Path | list[Any],
    model: Any | None = None,
    processor: Any | None = None,
    fps: int = 4,
) -> int:
    """
    Run rapid 1-token inference to detect if robot is about to pour.
    Returns 1 if about to pour, 0 otherwise.
    When COSMOS_REMOTE_URL is set, calls remote server instead of local model.
    """
    remote_url = os.environ.get("COSMOS_REMOTE_URL")
    if remote_url:
        created_temp = False
        if isinstance(video_path, (list, tuple)):
            video_path = _frames_to_video_path(video_path, fps=fps)
            created_temp = True
        try:
            return _binary_check_remote(remote_url, video_path, fps)
        finally:
            if created_temp and isinstance(video_path, Path) and video_path.exists():
                video_path.unlink(missing_ok=True)

    if model is None or processor is None:
        model, processor = load_cosmos_model()

    if isinstance(video_path, (list, tuple)):
        video_path = _frames_to_video_path(video_path, fps=fps)

    video_path = str(video_path)
    video_messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a lab assistant monitoring a robotic arm pouring liquids."}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": BINARY_PROMPT},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        video_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=5)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    text = (output_text[0] or "").strip().lower()
    # Parse 1/0 or yes/no (1 = about to pour)
    if "1" in text or text.startswith("yes"):
        return 1
    return 0


def _full_reason_remote(
    url: str,
    video_path: str | Path,
    prompt_path: str | Path,
    fps: int,
    max_new_tokens: int,
) -> str:
    """Call remote server for full reason. video_path must be a file path."""
    import httpx

    url = url.rstrip("/")
    with open(prompt_path, "r") as f:
        prompt = f.read()
    with open(video_path, "rb") as f:
        files = {"video": (Path(video_path).name, f, "video/mp4")}
        data = {"fps": fps, "prompt": prompt, "max_new_tokens": max_new_tokens}
        with httpx.Client(timeout=120.0) as client:
            r = client.post(f"{url}/full_reason", files=files, data=data)
            r.raise_for_status()
            return r.json().get("output", "")


def cosmos_full_reason(
    video_path: str | Path | list[Any],
    prompt_path: str | Path = "prompt.txt",
    model: Any | None = None,
    processor: Any | None = None,
    fps: int = 4,
    max_new_tokens: int = 4096,
) -> str:
    """
    Run full reasoning on video to determine if pouring trajectory is on track.
    Returns the model's reasoning output.
    When COSMOS_REMOTE_URL is set, calls remote server instead of local model.
    """
    remote_url = os.environ.get("COSMOS_REMOTE_URL")
    if remote_url:
        created_temp = False
        if isinstance(video_path, (list, tuple)):
            video_path = _frames_to_video_path(video_path, fps=fps)
            created_temp = True
        try:
            return _full_reason_remote(
                remote_url, video_path, Path(prompt_path), fps, max_new_tokens
            )
        finally:
            if created_temp and isinstance(video_path, Path) and video_path.exists():
                video_path.unlink(missing_ok=True)

    if model is None or processor is None:
        model, processor = load_cosmos_model()

    if isinstance(video_path, (list, tuple)):
        video_path = _frames_to_video_path(video_path, fps=fps)

    video_path = str(video_path)
    with open(prompt_path, "r") as f:
        prompt = f.read()

    video_messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a lab assistant monitoring a robotic arm pouring liquids."}]},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path, "fps": fps},
                {"type": "text", "text": prompt},
            ],
        },
    ]

    inputs = processor.apply_chat_template(
        video_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    output = output_text[0] or ""
    logger.info("Cosmos full reason output: %s", output)
    return output


# Standalone script behavior (original reason.py)
if __name__ == "__main__":
    model, processor = load_cosmos_model()
    with open("prompt.txt", "r") as f:
        prompt = f.read()

    video_messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": "saved_videos/output_phone/episode_000.mp4",
                    "fps": 4,
                },
                {
                    "type": "video",
                    "video": "saved_videos/output/episode_000.mp4",
                    "fps": 4,
                },
                {"type": "text", "text": prompt},
            ],
        },
    ]

    start = time.time()
    inputs = processor.apply_chat_template(
        video_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=4,
    )
    preprocess_time = time.time()
    print("Time to preprocess: " + str(preprocess_time - start))
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids, strict=False)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    print(output_text)
    print("Total time: " + str(time.time() - start))
