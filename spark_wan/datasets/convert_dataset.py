import json
import os
from pathlib import Path

import cv2
from tqdm import tqdm


root_path = "/mnt/workspace/checkpoints/finetrainers/3dgs-dissolve/"
video_txt = f"{root_path}/videos.txt"
prompt_txt = f"{root_path}/prompt.txt"
output_json = "./video_prompt.json"

data_list = []

with open(video_txt, "r") as vf, open(prompt_txt, "r") as pf:
    video_paths = [line.strip() for line in vf if line.strip()]
    prompts = [line.strip() for line in pf if line.strip()]

assert len(video_paths) == len(prompts), "video.txt 和 prompt.txt 行数不一致"

for video_path, prompt in tqdm(zip(video_paths, prompts), total=len(video_paths), desc="Processing videos"):
    video_path_obj = Path(os.path.join(root_path, video_path))
    if not video_path_obj.exists():
        print(f"⚠️ Video not found: {video_path}")
        continue

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        print(f"⚠️ Failed to open video: {video_path}")
        continue

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    item = {
        "path": str(video_path_obj).replace(root_path, ""),
        "cap": [prompt],
        "cut": [0, num_frames - 1],
        "crop": [0, width, 0, height],
        "resolution": {"width": width, "height": height},
        "num_frames": num_frames,
    }
    data_list.append(item)
    cap.release()

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(data_list, f, ensure_ascii=False, indent=2)

print(f"✅ Successfully saved {len(data_list)} items to {output_json}")
