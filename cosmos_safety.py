"""
Cosmos Safety Monitor for LeRobot SO101.

Detects "about to pour" moments with a fast 1-token binary check,
pauses the robot when detected, then runs full reasoning to validate trajectory.
"""

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Import reason module - use same directory
try:
    from reason import (
        cosmos_binary_check,
        cosmos_full_reason,
        load_cosmos_model,
    )
except ImportError:
    try:
        from cosmos.reason import (
            cosmos_binary_check,
            cosmos_full_reason,
            load_cosmos_model,
        )
    except ImportError:
        cosmos_binary_check = None
        cosmos_full_reason = None
        load_cosmos_model = None


class FrameBuffer:
    """Thread-safe buffer holding last N frames per camera for Cosmos video input."""

    def __init__(
        self,
        max_frames: int = 32,
        sample_rate: int = 8,
        camera_keys: list[str] | None = None,
    ):
        """
        Args:
            max_frames: Max frames to keep per camera (at 30 fps, 32 ~ 1 sec).
            sample_rate: Sample every Nth frame to get ~4 fps (30/8 ~ 4).
            camera_keys: Keys to treat as cameras in raw_observation. If None, auto-detect.
        """
        self.max_frames = max_frames
        self.sample_rate = sample_rate
        self.camera_keys = camera_keys
        self._buffers: dict[str, deque] = {}
        self._frame_count = 0
        self._lock = threading.Lock()

    def _get_camera_keys(self, raw_observation: dict) -> list[str]:
        if self.camera_keys is not None:
            return [k for k in self.camera_keys if k in raw_observation]
        # Auto-detect: keys whose values are numpy arrays with 3 dims (H, W, C)
        import numpy as np

        keys = []
        for k, v in raw_observation.items():
            if k in ("task",) or k.endswith(".pos"):
                continue
            if isinstance(v, np.ndarray) and v.ndim == 3:
                keys.append(k)
        return keys

    def push(self, raw_observation: dict[str, Any], camera_key: str | None = None) -> None:
        """Push frames from raw_observation into the buffer.

        Args:
            raw_observation: Dict of sensor readings.
            camera_key: If provided, only buffer this single camera key.
        """
        all_camera_keys = self._get_camera_keys(raw_observation)
        if camera_key is not None:
            if camera_key not in raw_observation:
                # Log once when the requested key isn't found
                if not getattr(self, '_logged_missing_key', False):
                    logger.warning(
                        f"Requested camera_key={camera_key!r} not found in observation. "
                        f"Available camera keys: {all_camera_keys}. "
                        f"All observation keys: {list(raw_observation.keys())}"
                    )
                    self._logged_missing_key = True
                camera_keys = []
            else:
                camera_keys = [camera_key]
        else:
            camera_keys = all_camera_keys
        # Log the first time we successfully buffer
        if camera_keys and not getattr(self, '_logged_camera_keys', False):
            logger.info(f"FrameBuffer: buffering camera key(s): {camera_keys} (all detected: {all_camera_keys})")
            self._logged_camera_keys = True
        with self._lock:
            self._frame_count += 1
            if self._frame_count % self.sample_rate != 0:
                return
            for key in camera_keys:
                frame = raw_observation[key]
                if key not in self._buffers:
                    self._buffers[key] = deque(maxlen=self.max_frames)
                self._buffers[key].append(frame.copy())

    def get_clip(self, camera_key: str | None = None) -> list[Any]:
        """
        Get a clip of frames for Cosmos. If camera_key is None, use first available.
        Returns list of numpy arrays (H, W, 3).
        """
        with self._lock:
            keys = list(self._buffers.keys())
            if not keys:
                return []
            key = camera_key if camera_key and camera_key in self._buffers else keys[0]
            return list(self._buffers[key])

    def get_combined_clip(self) -> list[Any]:
        """
        Get a clip combining all camera feeds side-by-side.
        Frames from each camera are horizontally concatenated.
        Returns list of numpy arrays (H, W_total, 3).
        """
        import numpy as np

        with self._lock:
            keys = sorted(self._buffers.keys())
            if not keys:
                return []
            # Use the shortest buffer length so all cameras are aligned
            min_len = min(len(self._buffers[k]) for k in keys)
            if min_len == 0:
                return []
            combined = []
            for i in range(min_len):
                frames_to_concat = []
                target_h = None
                for k in keys:
                    frame = self._buffers[k][i]
                    if target_h is None:
                        target_h = frame.shape[0]
                    # Resize to match height if needed
                    if frame.shape[0] != target_h:
                        from PIL import Image
                        img = Image.fromarray(frame)
                        new_w = int(frame.shape[1] * target_h / frame.shape[0])
                        img = img.resize((new_w, target_h))
                        frame = np.array(img)
                    frames_to_concat.append(frame)
                combined.append(np.concatenate(frames_to_concat, axis=1))
            return combined

    def clear(self) -> None:
        """Clear all buffered frames."""
        with self._lock:
            self._buffers.clear()
            self._frame_count = 0

    def has_enough_frames(self, min_frames: int = 8) -> bool:
        """Check if we have enough frames for a clip."""
        with self._lock:
            for buf in self._buffers.values():
                if len(buf) >= min_frames:
                    return True
            return False


class CosmosBinaryChecker:
    """Runs rapid 1-token Cosmos inference to detect 'about to pour'."""

    def __init__(self, model: Any = None, processor: Any = None):
        self.model = model
        self.processor = processor
        # Skip local model loading when using remote server
        if os.environ.get("COSMOS_REMOTE_URL"):
            return
        if model is None or processor is None:
            if load_cosmos_model is None:
                raise RuntimeError("Cosmos reason module not available")
            self.model, self.processor = load_cosmos_model()

    def check(self, frames: list[Any], fps: int = 4) -> int:
        """Returns 1 if about to pour, 0 otherwise."""
        if not frames:
            return 0
        return cosmos_binary_check(
            frames,
            model=self.model,
            processor=self.processor,
            fps=fps,
        )


class CosmosFullReasoner:
    """Runs full Cosmos reasoning to validate pouring trajectory."""

    def __init__(
        self,
        model: Any = None,
        processor: Any = None,
        prompt_path: str | Path = "prompt.txt",
    ):
        self.model = model
        self.processor = processor
        self.prompt_path = Path(prompt_path)
        # Skip local model loading when using remote server
        if os.environ.get("COSMOS_REMOTE_URL"):
            return
        if model is None or processor is None:
            if load_cosmos_model is None:
                raise RuntimeError("Cosmos reason module not available")
            self.model, self.processor = load_cosmos_model()

    def reason(self, frames: list[Any], fps: int = 4, max_new_tokens: int = 512) -> str:
        """Run full reasoning on the clip. Returns model output string."""
        if not frames:
            return ""
        return cosmos_full_reason(
            frames,
            prompt_path=self.prompt_path,
            model=self.model,
            processor=self.processor,
            fps=fps,
            max_new_tokens=max_new_tokens,
        )


def _parse_reason_output(output: str) -> bool:
    """
    Parse Cosmos full reason output to determine if trajectory is on track.
    Returns True if safe to resume, False if should abort.
    Expects 1 (on track) or 0 (not on track) in the output.
    """
    output_lower = output.lower().strip()

    # Look for 1 or 0 (prefer last occurrence, after </think>)
    if "</think>" in output_lower:
        after_think = output_lower.split("</think>")[-1].strip()
        tail = after_think[-50:] if len(after_think) > 50 else after_think
        if "1" in tail and "0" not in tail:
            return True
        if "0" in tail:
            return False

    # Fallback: 1 = resume, 0 = abort
    if "1" in output_lower and "0" not in output_lower:
        return True
    if "0" in output_lower:
        return False

    # Default: abort if unclear
    return False


class CosmosSafetyMonitor:
    """
    Orchestrates binary check, pause, and full reasoning.
    Runs binary check in a background thread; triggers full reason when paused.
    """

    def __init__(
        self,
        frame_buffer: FrameBuffer,
        binary_checker: CosmosBinaryChecker,
        full_reasoner: CosmosFullReasoner,
        binary_check_interval: float = 1.0,
        min_frames_for_check: int = 8,
        camera_key: str | None = None,
        prompt_path: str | Path = "prompt.txt",
    ):
        self.frame_buffer = frame_buffer
        self.binary_checker = binary_checker
        self.full_reasoner = full_reasoner
        self.binary_check_interval = binary_check_interval
        self.min_frames_for_check = min_frames_for_check
        self.camera_key = camera_key
        self.prompt_path = Path(prompt_path)

        self._pause_flag = False
        self._pause_lock = threading.Lock()
        self._shutdown = threading.Event()
        self._binary_thread: threading.Thread | None = None
        self._reasoning = False

    @property
    def is_paused(self) -> bool:
        with self._pause_lock:
            return self._pause_flag

    def _set_paused(self, value: bool) -> None:
        with self._pause_lock:
            self._pause_flag = value

    def push_observation(self, raw_observation: dict[str, Any]) -> None:
        """Call from control_loop_observation to buffer frames."""
        self.frame_buffer.push(raw_observation, camera_key=self.camera_key)

    def _run_binary_check_loop(self) -> None:
        """Background thread: run binary check periodically."""
        logger.info("Binary check loop started")
        last_check = 0.0
        while not self._shutdown.is_set():
            time.sleep(0.5)  # Wake every 0.5 sec
            if self._shutdown.is_set():
                break
            now = time.monotonic()
            if now - last_check < self.binary_check_interval:
                continue
            last_check = now
            if not self.frame_buffer.has_enough_frames(self.min_frames_for_check):
                with self.frame_buffer._lock:
                    buf_sizes = {k: len(v) for k, v in self.frame_buffer._buffers.items()}
                logger.info(f"Not enough frames yet. Buffers: {buf_sizes}, need {self.min_frames_for_check}")
                continue
            if self._reasoning:
                continue
            try:
                frames = self.frame_buffer.get_clip(self.camera_key)
                if len(frames) < self.min_frames_for_check:
                    continue
                logger.info(f"Running binary check with {len(frames)} frames")
                result = self.binary_checker.check(frames, fps=4)
                logger.info(f"Binary check result: {result}")
                if result == 1:
                    logger.info("Cosmos binary check: about to pour detected - pausing robot")
                    self._set_paused(True)
                    # Run full reason in background to decide resume/abort
                    reason_thread = threading.Thread(
                        target=self._run_full_reason_and_apply,
                        daemon=True,
                    )
                    reason_thread.start()
            except Exception as e:
                logger.warning(f"Cosmos binary check failed: {e}")

    def start(self) -> None:
        """Start the background binary check thread. Safe to call multiple times."""
        # Reset state from previous episode
        self._set_paused(False)
        self._reasoning = False
        self.frame_buffer.clear()
        self._shutdown.clear()
        self._binary_thread = threading.Thread(target=self._run_binary_check_loop, daemon=True)
        self._binary_thread.start()
        logger.info("CosmosSafetyMonitor started")

    def stop(self) -> None:
        """Stop the background thread."""
        self._shutdown.set()
        if self._binary_thread is not None:
            self._binary_thread.join(timeout=2.0)
            self._binary_thread = None

    def _run_full_reason_and_apply(self) -> None:
        """Run full reason (called from background thread when pause triggered)."""
        try:
            resume = self.run_full_reason_and_decide()
            if resume:
                self.resume()
            else:
                self.abort()
        except Exception as e:
            logger.warning(f"Cosmos full reason failed: {e} - remaining paused")

    def run_full_reason_and_decide(self) -> bool:
        """
        Run full reasoning (blocking). Call when paused.
        Returns True to resume, False to abort.
        """
        self._reasoning = True
        try:
            frames = self.frame_buffer.get_clip(self.camera_key)
            if not frames:
                logger.warning("No frames for full reason - resuming")
                return True
            logger.info(f"Running full reason with {len(frames)} frames")
            output = self.full_reasoner.reason(frames, fps=4)
            logger.info(f"Cosmos full reason output: {output}")
            resume = _parse_reason_output(output)
            logger.info(f"Full reason decision: {'resume' if resume else 'abort'}")
            return resume
        finally:
            self._reasoning = False

    def resume(self) -> None:
        """Clear pause flag after full reason says OK."""
        self._set_paused(False)
        logger.info("CosmosSafetyMonitor: resuming robot")

    def abort(self) -> None:
        """Keep paused (or require human intervention)."""
        logger.warning("CosmosSafetyMonitor: trajectory not on track - remaining paused")
