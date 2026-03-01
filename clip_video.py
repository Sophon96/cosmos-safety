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
) -> Path:
    """Clip video from start_sec to end_sec and save to output_dir."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Output filename: original_name_clip_1-10.mp4
    output_name = f"{video_path.stem}_clip_{start_sec}-{end_sec}.mp4"
    output_path = output_dir / output_name

    ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
    duration = end_sec - start_sec

    # -ss before -i = fast seek (input seeking)
    # -t = duration to copy
    cmd = [
        ffmpeg_path,
        "-y",  # overwrite output
        "-ss", str(start_sec),
        "-i", str(video_path),
        "-t", str(duration),
        "-c", "copy",  # stream copy, no re-encoding
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

    args = parser.parse_args()

    if args.start >= args.end:
        parser.error("start must be less than end")

    output_path = clip_video(
        args.video_path,
        args.start,
        args.end,
        args.output_dir,
    )
    print(f"Saved clipped video to: {output_path}")


if __name__ == "__main__":
    main()
