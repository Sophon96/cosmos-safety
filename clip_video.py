#!/usr/bin/env python3
"""Clip a video to a specified time range and save to videos/temp/."""

import argparse
import subprocess
from pathlib import Path

import imageio_ffmpeg


def clip_video(
    video_path: str,
    start_sec: float,
    end_sec: float,
    output_dir: str | Path = "videos/temp",
    fps: int | None = None,
    duration: float | None = None,
) -> Path:
    """Clip video from start_sec to end_sec and save to output_dir.

    If fps or duration is specified, re-encodes to produce exactly that many
    frames (e.g., fps=4, duration=1 -> 4 frames total). Otherwise uses
    stream copy for fast clipping without re-encoding.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    clip_duration = end_sec - start_sec

    # Output filename
    if fps is not None or duration is not None:
        out_dur = duration if duration is not None else clip_duration
        out_fps = fps if fps is not None else 4
        output_name = f"{video_path.stem}_clip_{start_sec}-{end_sec}_{out_fps}fps_{out_dur}s.mp4"
    else:
        output_name = f"{video_path.stem}_clip_{start_sec}-{end_sec}.mp4"
    output_path = output_dir / output_name

    if fps is not None or duration is not None:
        # Re-encode to set fps and/or duration
        out_duration = duration if duration is not None else clip_duration
        out_fps = fps if fps is not None else 4
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(out_duration),
            "-vf", f"fps={out_fps}",
            "-c:v", "libx264",
            "-crf", "23",
            "-an",
            str(output_path),
        ]
    else:
        # Stream copy, no re-encoding
        cmd = [
            ffmpeg_path,
            "-y",
            "-ss", str(start_sec),
            "-i", str(video_path),
            "-t", str(clip_duration),
            "-c", "copy",
            str(output_path),
        ]

    subprocess.run(cmd, check=True)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Clip a video to a time range (e.g., seconds 1-10)"
    )
    parser.add_argument(
        "video_path",
        help="Path to the input video",
    )
    parser.add_argument(
        "start",
        type=float,
        help="Start time in seconds",
    )
    parser.add_argument(
        "end",
        type=float,
        help="End time in seconds",
    )
    parser.add_argument(
        "-o", "--output-dir",
        default="videos/temp",
        help="Output directory (default: videos/temp)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Output frame rate (re-encodes; e.g., 4 for 4 fps)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Output duration in seconds (re-encodes; e.g., 1 for 1 second)",
    )

    args = parser.parse_args()

    if args.start >= args.end:
        parser.error("start must be less than end")

    output_path = clip_video(
        args.video_path,
        args.start,
        args.end,
        args.output_dir,
        fps=args.fps,
        duration=args.duration,
    )
    print(f"Saved clipped video to: {output_path}")


if __name__ == "__main__":
    main()
