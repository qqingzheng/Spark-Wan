import json
import logging
import os
import pickle
from typing import List, Optional, Tuple

import decord
import torch
import torch.distributed as dist
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo, ToTensorVideo
from tqdm import tqdm

from diffusers.training_utils import free_memory


class EasyVideoDataset(Dataset):
    def __init__(
        self,
        video_data: List[Tuple[str, str]],
        video_size: Tuple[int, int] = (720, 1280),
        num_frames: int = 81,
        cache_dir: Optional[str] = "./dataset_cache",
        skip_frames_start: int = 0,
        skip_frames_end: int = 0,
        id_token: Optional[str] = None,
    ):
        self.video_data = video_data
        self.video_size = video_size
        self.num_frames = num_frames
        self.skip_frames_start = skip_frames_start
        self.skip_frames_end = skip_frames_end
        self.id_token = id_token or ""

        self.transform_1 = Compose(
            [
                ToTensorVideo(),
                CenterCropVideo(self.video_size),
                # NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )
        self.transform_2 = Compose(
            [
                # ToTensorVideo(),
                # CenterCropVideo(self.video_size),
                NormalizeVideo(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
            ]
        )

        self.data = []

        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir, exist_ok=True)

        for base_dir, json_or_pkl_path in tqdm(video_data, desc="Processing video data"):
            file_base_name = (
                os.path.basename(json_or_pkl_path)
                + f"_{self.num_frames}_{self.video_size[0]}_{self.video_size[1]}.cache"
            )
            cache_path = os.path.join(cache_dir, file_base_name)
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as fi:
                    self.data.extend(pickle.load(fi))
                continue

            if json_or_pkl_path.endswith(".json"):
                with open(json_or_pkl_path, "r") as fi:
                    data = json.load(fi)
            else:
                with open(json_or_pkl_path, "rb") as fi:
                    data = pickle.load(fi)
            cache_data = []
            for item in tqdm(data, desc="Processing items in file", leave=False):
                video_path = os.path.join(base_dir, item["path"])
                if not os.path.exists(video_path):
                    continue

                # if (
                #     item["resolution"]["width"] < self.video_size[1]
                #     or item["resolution"]["height"] < self.video_size[0]
                # ):
                #     continue
                if item["num_frames"] < self.num_frames:
                    continue

                cache_data.append(
                    {
                        "path": video_path,
                        "caption": self.id_token + item["cap"][0],
                        "cut": item["cut"],
                        "crop": item["crop"],
                    }
                )
            print(f"{base_dir} has {len(cache_data)} available videos")
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"Saving cache to {cache_path}")
                with open(cache_path, "wb") as fi:
                    pickle.dump(cache_data, fi)
            self.data.extend(cache_data)

        logging.info(f"Totoal data: {len(self.data)}!")

    def save_video(self, video_tensor, save_path="./check.mp4", fps=30):
        import imageio

        video_np = video_tensor.cpu().numpy()  # ensure on CPU
        writer = imageio.get_writer(save_path, fps=fps, codec="libx264")

        for frame in video_np:
            writer.append_data(frame)
        writer.close()

    def resize_to_fit_crop(self, video, target_size):
        t, h, w, c = video.shape
        target_h, target_w = target_size

        scale = max(target_h / h, target_w / w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))

        # (T, H, W, C) -> (T, C, H, W)
        video = video.permute(0, 3, 1, 2).float()
        video = F.interpolate(video, size=(new_h, new_w), mode="bilinear", align_corners=False)
        # (T, C, H, W) -> (T, H, W, C)
        video = video.permute(0, 2, 3, 1)
        video = video.clamp(0, 255).to(torch.uint8)

        return video

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_reader = decord.VideoReader(self.data[idx]["path"])

        start = self.data[idx]["cut"][0]
        video = video_reader.get_batch(list(range(start, start + self.num_frames))).asnumpy()
        video = torch.from_numpy(video)  # t h w c
        video = video[
            :,
            self.data[idx]["crop"][2] : self.data[idx]["crop"][3],
            self.data[idx]["crop"][0] : self.data[idx]["crop"][1],
            :,
        ]
        video = self.resize_to_fit_crop(video=video, target_size=self.video_size)
        video = self.transform_1(video)
        first_frame = video[:, 0:1]
        video = self.transform_2(video)  # c t h w

        # Squeeze the temporal dimension to get image
        first_frame = first_frame.squeeze(1).permute(1, 2, 0)
        first_frame = (first_frame * 255).clamp(0, 255).to(torch.uint8)
        image = Image.fromarray(first_frame.numpy())

        del video_reader
        free_memory()

        # import imageio.v2 as imageio
        # save_dir = "./saved_videos"
        # os.makedirs(save_dir, exist_ok=True)
        # save_path = os.path.join(save_dir, f"video_{idx}.mp4")
        # video_np = video.permute(1, 2, 3, 0).cpu().numpy()  # t h w c
        # writer = imageio.get_writer(save_path, fps=15)
        # for frame in video_np:
        #     writer.append_data(frame)
        # writer.close()

        # try:
        #     video_reader = decord.VideoReader(self.data[idx]["path"])

        #     start = self.data[idx]["cut"][0]
        #     video = video_reader.get_batch(
        #         list(range(start, start + self.num_frames))
        #     ).asnumpy()
        #     video = torch.from_numpy(video)  # t h w c
        #     video = video[
        #         :,
        #         self.data[idx]["crop"][2] : self.data[idx]["crop"][3],
        #         self.data[idx]["crop"][0] : self.data[idx]["crop"][1],
        #         :,
        #     ]
        #     video = self.resize_shorter_side(video=video, target_shorter=self.video_size[0] if self.video_size[0] < self.video_size[1] else self.video_size[1])
        #     video = self.transform(video)  # c t h w

        #     # Get first frame as image
        #     first_frame = video_reader.get_batch([start]).asnumpy()
        #     first_frame = torch.from_numpy(first_frame)  # 1 h w c
        #     first_frame = first_frame[
        #         :,
        #         self.data[idx]["crop"][2] : self.data[idx]["crop"][3],
        #         self.data[idx]["crop"][0] : self.data[idx]["crop"][1],
        #         :,
        #     ]
        #     # Squeeze the temporal dimension to get image
        #     image = first_frame.squeeze(0)
        #     image = Image.fromarray(image.numpy())

        #     del video_reader
        #     free_memory()

        # except Exception:  # If loading video failed, return a random video
        #     print("error load data")
        #     return self.__getitem__(random.randint(0, len(self.data) - 1))

        return {
            "instance_video": video,
            "instance_prompt": self.data[idx]["caption"],
            "instance_image": image,
        }


if __name__ == "__main__":
    from PIL import Image

    height = 480
    width = 832
    max_num_frames = 73
    instance_data_root = (
        "/mnt/workspace/ysh/Code/Efficient_Model/2_code/framepack_wan/dance_forcing/dataset/Disney_temp_final.json"
    )
    weight_dtype = torch.bfloat16
    device = "cuda"

    def easy_collate_fn(examples):
        videos = [example["instance_video"].unsqueeze(0) for example in examples]
        prompts = [example["instance_prompt"] for example in examples]
        images = [example["instance_image"] for example in examples]
        videos = torch.cat(videos, dim=0)
        return {
            "videos": videos,
            "prompts": prompts,
            "images": images,
        }

    train_dataset = EasyVideoDataset(
        video_size=(height, width),
        num_frames=max_num_frames,
        video_data=json.load(open(instance_data_root)),
        # id_token="DISNEY"
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0,
        collate_fn=easy_collate_fn,
    )

    import sys

    from transformers import AutoTokenizer, CLIPImageProcessor, CLIPVisionModel, UMT5EncoderModel

    from diffusers.models import AutoencoderKLWan
    from diffusers.video_processor import VideoProcessor

    sys.path.append("../")
    from utils.utils_framepack import (
        encode_image_1,
        encode_image_2,
        encode_prompt,
        get_framepack_input,
    )

    latent_window_size = 9
    is_vanilla_sampling = False
    pretrained_model_name_or_path = "/mnt/workspace/checkpoints/Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
    # load tokenizers
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="tokenizer",
    )
    image_processor = CLIPImageProcessor.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="image_processor",
    )
    # load encoders
    text_encoder = UMT5EncoderModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="text_encoder",
        torch_dtype=torch.float16,
    )
    image_encoder = CLIPVisionModel.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="image_encoder",
        torch_dtype=torch.float16,
    )
    # load vae
    vae = AutoencoderKLWan.from_pretrained(
        pretrained_model_name_or_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    vae_scale_factor_temporal = 2 ** sum(vae.temperal_downsample)
    vae_scale_factor_spatial = 2 ** len(vae.temperal_downsample)
    video_processor = VideoProcessor(vae_scale_factor=vae_scale_factor_spatial)
    assert max_num_frames % vae_scale_factor_temporal == 1

    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)
    vae.eval()
    text_encoder.eval()
    image_encoder.eval()
    text_encoder.to(device)
    image_encoder.to(device)
    vae.to(device)

    print("Testing dataloader...")
    for i, batch in enumerate(train_dataloader):
        print(f"Batch {i}:")

        pixel_values = batch["videos"].to(device, dtype=weight_dtype)
        prompts = batch["prompts"]
        images = batch["images"]

        # Prepare videos
        (
            model_input,  # torch.Size([2, 16, 9, 60, 104])
            indices_latents,  # torch.Size([2, 9])
            latents_clean,  # torch.Size([2, 16, 2, 60, 104])
            indices_clean_latents,  # torch.Size([2, 2])
            latents_history_2x,  # torch.Size([2, 16, 2, 60, 104])
            indices_latents_history_2x,  # torch.Size([2, 2])
            latents_history_4x,  # torch.Size([2, 16, 16, 60, 104])
            indices_latents_history_4x,  # torch.Size([2, 16])
            section_to_video_idx,
        ) = get_framepack_input(
            vae=vae,
            pixel_values=pixel_values,  # torch.Size([1, 3, 73, 480, 832])
            latent_window_size=latent_window_size,  # 9
            vanilla_sampling=is_vanilla_sampling,
            dtype=weight_dtype,
        )

        # Encode prompts
        prompt_embeds = encode_prompt(
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            prompt=prompts,
            device=device,
            dtype=weight_dtype,
        )
        prompt_embeds = prompt_embeds.to(device, weight_dtype)

        # Prepare images
        image_embeds = encode_image_1(
            image_processor=image_processor, image_encoder=image_encoder, image=images, device=device
        )
        image_embeds = image_embeds.to(device, weight_dtype)

        image_conditions = video_processor.preprocess(images, height=height, width=width).to(
            device, dtype=torch.float32
        )
        image_conditions = encode_image_2(
            vae=vae,
            vae_scale_factor_temporal=vae_scale_factor_temporal,
            vae_scale_factor_spatial=vae_scale_factor_spatial,
            image=image_conditions,
            height=height,
            width=width,
            num_frames=(latent_window_size - 1) * vae_scale_factor_temporal + 1,
            dtype=weight_dtype,
            device=device,
        )
        image_conditions = image_conditions.to(device, weight_dtype)

        # Repeat
        section_to_video_idx = torch.tensor(
            section_to_video_idx,
            dtype=torch.long,
            device=device,
        )
        prompt_embeds = prompt_embeds[section_to_video_idx]
        image_embeds = image_embeds[section_to_video_idx]
        image_conditions = image_conditions[section_to_video_idx]

        import pdb

        pdb.set_trace()
