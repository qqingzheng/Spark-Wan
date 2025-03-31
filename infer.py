import sys

sys.path.append(".")

import os

import torch
import torch.distributed as dist
from diffusers import AutoencoderKLWan
from diffusers.schedulers import UniPCMultistepScheduler
from diffusers.utils import export_to_video
from diffusers import WanPipeline
from diffusers.image_processor import VaeImageProcessor
from spark_wan.models.transformer_wan import WanTransformer3DModel
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32
from spark_wan.parrallel.env import init_sequence_parallel_group
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from safetensors.torch import load_file
from tools.fp16_monitor import FP16Monitor

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())
sp_group_index, sp_group_local_rank = init_sequence_parallel_group(8)
weight_dtype = torch.bfloat16
seed = 2002
prompts = [
    "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.",
    "Several giant wooly mammoths approach treading through a snowy meadow, their long wooly fur lightly blows in the wind as they walk, snow covered trees and dramatic snow capped mountains in the distance, mid afternoon light with wispy clouds and a sun high in the distance creates a warm glow, the low camera view is stunning capturing the large furry mammal with beautiful photography, depth of field.",
    "A movie trailer featuring the adventures of the 30 year old space man wearing a red wool knitted motorcycle helmet, blue sky, salt desert, cinematic style, shot on 35mm film, vivid colors.",
    "Drone view of waves crashing against the rugged cliffs along Big Sur’s garay point beach. The crashing blue waters create white-tipped waves, while the golden light of the setting sun illuminates the rocky shore. A small island with a lighthouse sits in the distance, and green shrubbery covers the cliff’s edge. The steep drop from the road down to the beach is a dramatic feat, with the cliff’s edges jutting out over the sea. This is a view that captures the raw beauty of the coast and the rugged landscape of the Pacific Coast Highway.",
    "The camera follows behind a white vintage SUV with a black roof rack as it speeds up a steep dirt road surrounded by pine trees on a steep mountain slope, dust kicks up from it’s tires, the sunlight shines on the SUV as it speeds along the dirt road, casting a warm glow over the scene. The dirt road curves gently into the distance, with no other cars or vehicles in sight. The trees on either side of the road are redwoods, with patches of greenery scattered throughout. The car is seen from the rear following the curve with ease, making it seem as if it is on a rugged drive through the rugged terrain. The dirt road itself is surrounded by steep hills and mountains, with a clear blue sky above with wispy clouds."
]
negative_prompt = ""
# negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"
output_video_path = f"experiments/output/new_14B_8_distill_1000_bf16/"
state_dict_path = f"/storage/lcm/Wan-Distill/Spark/14B_16_8_bf16/checkpoint-1200/model.safetensors"

os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
vae = AutoencoderKLWan.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-14B-Diffusers/",
    subfolder="vae",
    torch_dtype=weight_dtype,
)
transformer = WanTransformer3DModel.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-14B-Diffusers/",
    subfolder="merged_model",
    torch_dtype=weight_dtype,
)
transformer = replace_rmsnorm_with_fp32(transformer)
lora_target_modules = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "proj_out",
    "ffn.net.0.proj",
    "ffn.net.2",
]
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=lora_target_modules
)
transformer = get_peft_model(transformer, lora_config)

state_dict = load_file(
    state_dict_path,
    device="cpu",
)
print(state_dict.keys())
missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)

scheduler = UniPCMultistepScheduler(
    prediction_type="flow_prediction",
    use_flow_sigmas=True,
    num_train_timesteps=1000,
    flow_shift=5.0,
)

pipe = WanPipeline.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-14B-Diffusers/",
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    torch_dtype=weight_dtype,
)

pipe = pipe.to(weight_dtype, device="cuda")

videos = []

idx = 0
for prompt in tqdm(prompts):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    with torch.cuda.amp.autocast(dtype=weight_dtype):
        pt_images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=720,
            width=1280,
            num_inference_steps=8,
            num_frames=81,
            guidance_scale=0.0,
            generator=generator,
            output_type="pt",
        ).frames[0]
    pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])
    image_np = VaeImageProcessor.pt_to_numpy(pt_images)
    image_pil = VaeImageProcessor.numpy_to_pil(image_np)
    if dist.get_rank() == 0:
        video_path = f"{output_video_path}/output_{idx}.mp4"
        export_to_video(image_pil, video_path, fps=16)
    idx += 1


dist.barrier()
