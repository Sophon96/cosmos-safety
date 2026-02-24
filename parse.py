"""Extract per-episode videos from record-pour-water using LeRobotDataset for correct alignment."""

from pathlib import Path
from collections import defaultdict

import imageio
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

output_dir = Path("output_phone")
output_dir.mkdir(exist_ok=True)

# Switch to "observation.images.phone" for phone camera
CAMERA = "observation.images.phone"

# LeRobotDataset handles parquet + video alignment via meta/episodes
# video_backend="pyav" avoids torchcodec/FFmpeg lib loading issues on macOS
REPO_ID = "Sophon96/record-pour-water"
local_path = Path("record-pour-water")
if local_path.exists() and (local_path / "meta").exists():
    ds = LeRobotDataset(str(local_path.absolute()), video_backend="pyav")
else:
    ds = LeRobotDataset(REPO_ID, video_backend="pyav")

# Group frames by episode using correctly aligned images from LeRobotDataset
episodes = defaultdict(list)
for i in range(len(ds)):
    sample = ds[i]
    ep = int(sample["episode_index"].item() if hasattr(sample["episode_index"], "item") else sample["episode_index"])
    img = sample[CAMERA]

    # Convert tensor to numpy uint8
    if hasattr(img, "numpy"):
        img = img.numpy()
    elif hasattr(img, "cpu"):
        img = img.cpu().numpy()
    else:
        img = np.array(img)

    if img.dtype != np.uint8:
        if img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    # LeRobotDataset returns [C,H,W], imageio expects [H,W,C]
    if img.ndim == 3 and img.shape[0] in (1, 3):
        img = np.transpose(img, (1, 2, 0))
    if img.shape[-1] == 1:
        img = img.squeeze(-1)

    episodes[ep].append(img)

# Write one MP4 per episode
for ep in sorted(episodes.keys()):
    frames = episodes[ep]
    path = output_dir / f"episode_{ep:03d}.mp4"
    imageio.mimsave(str(path), frames, fps=30)
    print(f"Saved {path} ({len(frames)} frames)")
