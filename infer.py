import sys
sys.path.append(".")

import os

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import WanPipeline
from diffusers.image_processor import VaeImageProcessor
from spark_wan.models.transformer_wan import WanTransformer3DModel
from spark_wan.parrallel.env import init_sequence_parallel_group

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
sp_group_index, sp_group_local_rank = init_sequence_parallel_group(8)
weight_dtype = torch.float16
seed = 2002
prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
output_video_path = f"experiments/output/unipc_8_steps/"
os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

vae = AutoencoderKLWan.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-1.3B-Diffusers/", subfolder="vae", torch_dtype=weight_dtype
)
transformer = WanTransformer3DModel.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-1.3B-Diffusers/", subfolder="transformer", torch_dtype=weight_dtype
)

scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=8.0,
)

pipe = WanPipeline.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-1.3B-Diffusers/",
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=weight_dtype,
)

pipe = pipe.to(weight_dtype, device="cuda")

generator = torch.Generator(device="cuda").manual_seed(seed)
pt_images = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    height=480,
    width=832,
    num_inference_steps=8,
    num_frames=81,
    guidance_scale=5.0,
    generator=generator,
    output_type="pt"
).frames[0]
videos = []
pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
image_np = VaeImageProcessor.pt_to_numpy(pt_images)
image_pil = VaeImageProcessor.numpy_to_pil(image_np)
videos.append(image_pil)

if dist.get_rank() == 0:
    for idx, video in enumerate(videos):
        video_path = f"{output_video_path}/output_{idx}.mp4"
        export_to_video(video, video_path, fps=16)
dist.barrier()