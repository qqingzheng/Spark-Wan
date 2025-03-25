import sys
sys.path.append(".")

import argparse
import json
import os

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from peft import PeftModel

from models.pipeline_wan import WanPipeline
from models.transformer_wan_sp_lzj import WanTransformer3DModel
from util.utils import parse_partial_layer_idx
from models.sp_utils.env import init_sequence_parallel_group

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
sp_group_index, sp_group_local_rank = init_sequence_parallel_group(4)
weight_dtype = torch.float16
seed = 42
prompt = "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
output_path = f"experiments/remove_layer_test/"
os.makedirs(output_path, exist_ok=True)
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
    flow_shift=5.0 if transformer.config.num_layers == 40 else 8.0,
)

pipe = WanPipeline.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-1.3B-Diffusers/",
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=weight_dtype,
)

pipe = pipe.to(weight_dtype, device="cuda")

class DummyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

last_remove_block = None
for i in range(1, transformer.config.num_layers):
    last_remove_block = transformer.blocks[i]
    transformer.blocks[i] = DummyModule().to("cuda", dtype=weight_dtype)

    generator = torch.Generator(device="cuda").manual_seed(seed)
    output = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        height=480,
        width=832,
        num_inference_steps=32,
        num_frames=81,
        guidance_scale=4.0,
        generator=generator,
    ).frames[0]
    
    if dist.get_rank() == 0:
        video_path = f"{output_path}/{i}.mp4"
        export_to_video(output, video_path, fps=16)
    dist.barrier()
    transformer.blocks[i] = last_remove_block