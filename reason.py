# Unsloth must be imported before transformers for optimizations
# from unsloth import FastVisionModel
import torch
import transformers
import time
from pathlib import Path
from typing import Any

MODEL_NAME = "nvidia/Cosmos-Reason2-2B"
BINARY_PROMPT = (
    "Watch the video. Is the robot about to pour water from one cup into another? "
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


def cosmos_binary_check(
    video_path: str | Path | list[Any],
    model: Any | None = None,
    processor: Any | None = None,
    fps: int = 4,
) -> int:
    """
    Run rapid 1-token inference to detect if robot is about to pour.
    Returns 1 if about to pour, 0 otherwise.
    """
    if model is None or processor is None:
        model, processor = load_cosmos_model()

    if isinstance(video_path, (list, tuple)):
        video_path = _frames_to_video_path(video_path, fps=fps)

    video_path = str(video_path)
    video_messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
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


def cosmos_full_reason(
    video_path: str | Path | list[Any],
    prompt_path: str | Path = "prompt.txt",
    model: Any | None = None,
    processor: Any | None = None,
    fps: int = 4,
    max_new_tokens: int = 512,
) -> str:
    """
    Run full reasoning on video to determine if pouring trajectory is on track.
    Returns the model's reasoning output.
    """
    if model is None or processor is None:
        model, processor = load_cosmos_model()

    if isinstance(video_path, (list, tuple)):
        video_path = _frames_to_video_path(video_path, fps=fps)

    video_path = str(video_path)
    with open(prompt_path, "r") as f:
        prompt = f.read()

    video_messages = [
        {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
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

    return output_text[0] or ""


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
                    "video": "videos/snapshots/91ed530147ea8f380b10181c4e568865fa0e0996/output_phone/episode_049.mp4",
                    "fps": 4,
                },
                {
                    "type": "video",
                    "video": "videos/snapshots/91ed530147ea8f380b10181c4e568865fa0e0996/output/episode_049.mp4",
                    "fps": 4,
                },
                {"type": "text", "text": "what is happening in these two videos?"},
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
