import sys


sys.path.append(".")

import os

import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from spark_wan.models.transformer_wan import WanTransformer3DModel
from spark_wan.parrallel.env import init_sequence_parallel_group
from spark_wan.pipelines.pipeline_wan_t2v import WanPipeline
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32
from tqdm import tqdm

from diffusers import AutoencoderKLWan
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import export_to_video


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())


weight_dtype = torch.bfloat16
seed = 2002
flow_shift = 5.0
guidance_scale = 5.0
height = (720,)
width = (1280,)
num_inference_steps = (30,)
num_frames = 81

output_path = "output"
model_path = "/mnt/workspace/checkpoints/Wan-AI/Wan2.1-T2V-14B-Diffusers/"
lora_path = None
# lora_path = f"/data/pfs/checkpoints/chestnutlzj/Spark-Wan-16steps/model.safetensors"

prompts = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, along red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is dampand reflective, creating a mirror effect of thecolorful lights. Many pedestrians walk about."
]
negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"


sp_group_index, sp_group_local_rank = init_sequence_parallel_group(8)

os.makedirs(output_path, exist_ok=True)
vae = AutoencoderKLWan.from_pretrained(
    model_path,
    subfolder="vae",
    torch_dtype=torch.float32,
)
transformer = WanTransformer3DModel.from_pretrained(
    model_path,
    subfolder="transformer",
    torch_dtype=weight_dtype,
)
transformer = replace_rmsnorm_with_fp32(transformer)

if lora_path is not None:
    lora_target_modules = [
        "to_q",
        "to_k",
        "to_v",
        "to_out.0",
        "proj_out",
        "ffn.net.0.proj",
        "ffn.net.2",
    ]
    lora_config = LoraConfig(r=64, lora_alpha=128, target_modules=lora_target_modules)
    transformer = get_peft_model(transformer, lora_config)
    state_dict = load_file(
        lora_path,
        device="cpu",
    )
    missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
    print("unexpected_keys: ", unexpected_keys)
    print("loading lora done!")

transformer.eval()

# scheduler = UniPCMultistepScheduler(
#     prediction_type="flow_prediction",
#     use_flow_sigmas=True,
#     num_train_timesteps=1000,
#     flow_shift=flow_shift,
# )
scheduler = FlowMatchEulerDiscreteScheduler(
    num_train_timesteps=1000,
    shift=flow_shift,
)

pipe = WanPipeline.from_pretrained(
    model_path,
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=weight_dtype,
)

pipe = pipe.to(weight_dtype, device="cuda")

for prompt in tqdm(prompts):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            generator=generator,
            flash_attn=True,
        ).frames[0]
    if dist.get_rank() == 0:
        file_count = len([f for f in os.listdir(output_path) if os.path.isfile(os.path.join(output_path, f))])
        video_path = f"{output_path}/{seed}_{file_count:04d}.mp4"
        export_to_video(output, video_path, fps=16)

dist.barrier()
dist.destroy_process_group()
