import sys


sys.path.append(".")

import argparse
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
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler, UniPCMultistepScheduler
from diffusers.utils import export_to_video


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

dist.init_process_group(backend="nccl")
torch.cuda.set_device(dist.get_rank())


def parse_args():
    parser = argparse.ArgumentParser(description="Generate video with model")

    # Arguments for model parameters
    parser.add_argument(
        "--weight_dtype",
        type=str,
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Data type for model weights.",
    )
    parser.add_argument("--seed", type=int, default=2002, help="Seed for random number generator.")
    parser.add_argument("--flow_shift", type=float, default=5.0, help="Flow shift parameter.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale for model inference.")

    # Arguments for resolution
    parser.add_argument("--height", type=int, default=720, help="Height of output frames.")
    parser.add_argument("--width", type=int, default=1280, help="Width of output frames.")

    # Arguments for inference steps and frames
    parser.add_argument(
        "--scheduler_type",
        type=str,
        default="euler",
        choices=["unipc", "euler"],
    )
    parser.add_argument("--num_inference_steps", type=int, default=30, help="Number of inference steps.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of frames to generate.")

    # Arguments for paths
    parser.add_argument("--output_path", type=str, default="output", help="Directory to save output.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/mnt/workspace/checkpoints/Wan-AI/Wan2.1-T2V-14B-Diffusers/",
        help="Path to model directory.",
    )
    parser.add_argument(
        "--lora_path", type=str, default=None, help="Path to lora model (optional)."
    )  # "/data/pfs/checkpoints/chestnutlzj/Spark-Wan-16steps/model.safetensors"

    # Arguments for prompts
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=[
            "A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, along red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is dampand reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about."
        ],
        help="List of prompts for model generation.",
    )

    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
        help="Negative prompt for model generation.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.weight_dtype == "fp32":
        args.weight_dtype = torch.float32
    elif args.weight_dtype == "fp16":
        args.weight_dtype = torch.float16
    else:
        args.weight_dtype = torch.bfloat16

    sp_group_index, sp_group_local_rank = init_sequence_parallel_group(8)

    os.makedirs(args.output_path, exist_ok=True)
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
    )
    transformer = WanTransformer3DModel.from_pretrained(
        args.model_path,
        subfolder="transformer",
        torch_dtype=args.weight_dtype,
    )
    transformer = replace_rmsnorm_with_fp32(transformer)

    if args.lora_path is not None:
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
            args.lora_path,
            device="cpu",
        )
        missing_keys, unexpected_keys = transformer.load_state_dict(state_dict, strict=False)
        print("unexpected_keys: ", unexpected_keys)
        print("loading lora done!")

    transformer.eval()

    if args.scheduler_type == "unipc":
        scheduler = UniPCMultistepScheduler(
            prediction_type="flow_prediction",
            use_flow_sigmas=True,
            num_train_timesteps=1000,
            flow_shift=args.flow_shift,
        )
    else:
        scheduler = FlowMatchEulerDiscreteScheduler(
            num_train_timesteps=1000,
            shift=args.flow_shift,
        )

    pipe = WanPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        torch_dtype=args.weight_dtype,
    )

    pipe = pipe.to(args.weight_dtype, device="cuda")

    for prompt in tqdm(args.prompts):
        generator = torch.Generator(device="cuda").manual_seed(args.seed)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            output = pipe(
                prompt=prompt,
                negative_prompt=args.negative_prompt,
                height=args.height,
                width=args.width,
                num_inference_steps=args.num_inference_steps,
                num_frames=args.num_frames,
                guidance_scale=args.guidance_scale,
                generator=generator,
                is_flash_attn=True,
            ).frames[0]
        if dist.get_rank() == 0:
            file_count = len(
                [f for f in os.listdir(args.output_path) if os.path.isfile(os.path.join(args.output_path, f))]
            )
            video_path = f"{args.output_path}/{args.seed}_{file_count:04d}.mp4"
            export_to_video(output, video_path, fps=16)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
