import copy
import sys


sys.path.append(".")

import argparse
import logging
import math
import os
from typing import Optional

import torch
import torch.distributed as dist
import transformers
from spark_wan.parrallel.env import setup_sequence_parallel_group
from spark_wan.training_utils.fsdp2_utils import (
    load_model_state,
    load_optimizer_state,
    load_state,
    save_state,
    unwrap_model,
)
from spark_wan.training_utils.input_process import encode_prompt
from spark_wan.training_utils.load_dataset import load_easyvideo_dataset
from spark_wan.training_utils.load_model import (
    load_model,
)
from spark_wan.training_utils.load_optimizer import get_optimizer
from spark_wan.training_utils.train_config import Args
from torch.amp import GradScaler
from tqdm.auto import tqdm

import diffusers
from diffusers import (
    FlowMatchEulerDiscreteScheduler,
    WanPipeline,
)
from diffusers.image_processor import VaeImageProcessor
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
    set_seed,
)
from diffusers.utils import check_min_version, export_to_video, is_wandb_available


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")


def setup_distributed_env():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))


def cleanup_distributed_env():
    dist.destroy_process_group()


def main(args: Args):
    setup_distributed_env()

    global_rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.cuda.current_device()
    world_size = dist.get_world_size()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    if local_rank == 0:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # Handle the repository creation
    if global_rank == 0:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    # Mixed precision training
    weight_dtype = torch.float32
    if args.training_config.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.training_config.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    elif args.training_config.mixed_precision == "no":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Load model
    tokenizer, text_encoder, transformer, vae = load_model(
        pretrained_model_name_or_path=args.model_config.pretrained_model_name_or_path,
        compile_transformer=args.model_config.compile_transformer,
        fsdp_transformer=args.model_config.fsdp_transformer,
        fsdp_text_encoder=args.model_config.fsdp_text_encoder,
        weight_dtype=weight_dtype,
        device=device,
        is_train_lora=args.model_config.is_train_lora,
        lora_rank=args.model_config.lora_rank,
        lora_alpha=args.model_config.lora_alpha,
        lora_dropout=args.model_config.lora_dropout,
        pretrained_lora_path=args.model_config.pretrained_lora_path,
        find_unused_parameters=True,
        reshard_after_forward=args.parallel_config.reshard_after_forward,
    )

    # Setup scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler(
        num_train_timesteps=1000, shift=args.model_config.flow_shift
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)

    # Resume model state from checkpoint
    if args.training_config.resume_from_checkpoint:
        load_model_state(
            unwrap_model(transformer),
            args.training_config.resume_from_checkpoint,
            is_fsdp=args.model_config.fsdp_transformer,
            device=device,
            fsdp_cpu_offload=False,
        )

    # Setup optimizer
    transformer_lora_parameters = list(
        filter(lambda p: p.requires_grad, transformer.parameters())
    )
    transformer_parameters_with_lr = {
        "params": transformer_lora_parameters,
        "lr": args.training_config.learning_rate,
    }
    params_to_optimize = [transformer_parameters_with_lr]
    optimizer = get_optimizer(
        optimizer=args.training_config.optimizer,
        learning_rate=args.training_config.learning_rate,
        adam_beta1=args.training_config.adam_beta1,
        adam_beta2=args.training_config.adam_beta2,
        adam_epsilon=args.training_config.adam_epsilon,
        adam_weight_decay=args.training_config.adam_weight_decay,
        params_to_optimize=params_to_optimize,
    )

    # Resume optimizer state from checkpoint
    if args.training_config.resume_from_checkpoint:
        load_optimizer_state(
            optimizer,
            args.training_config.resume_from_checkpoint,
            is_fsdp=args.model_config.fsdp_transformer,
        )

    # Setup gradient scaler
    scaler = GradScaler()

    # Setup sequence parallel group
    sp_group_index, sp_group_local_rank, dp_rank, dp_size = (
        setup_sequence_parallel_group(args.parallel_config.sp_size)
    )
    set_seed(args.seed + dp_rank)

    # Load dataset
    train_dataloader, sampler = load_easyvideo_dataset(
        height=args.data_config.height,
        width=args.data_config.width,
        max_num_frames=args.data_config.max_num_frames,
        instance_data_root=args.data_config.instance_data_root,
        train_batch_size=args.training_config.train_batch_size,
        dataloader_num_workers=args.data_config.dataloader_num_workers,
        dp_rank=dp_rank,
        dp_size=dp_size,
    )

    # Initialize tracker
    if global_rank == 0:
        wandb.init(
            project=args.report_to.project_name,
            name=args.report_to.wandb_name,
            notes=args.report_to.wandb_notes,
            sync_tensorboard=True,
        )
        wandb.config.update(OmegaConf.to_container(args, resolve=True))

    # Scheduler.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if args.training_config.max_train_steps is None:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.training_config.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.training_config.lr_warmup_steps * world_size,
        num_training_steps=args.training_config.max_train_steps * world_size,
        num_cycles=args.training_config.lr_num_cycles,
        power=args.training_config.lr_power,
    )

    # Resume state from checkpoint
    if args.training_config.resume_from_checkpoint:
        global_step = load_state(
            args.training_config.resume_from_checkpoint,
            dataloader=train_dataloader,
            sampler=sampler,
            scaler=scaler,
            lr_scheduler=lr_scheduler,
        )
    else:
        global_step = 0

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.training_config.gradient_accumulation_steps
    )
    if overrode_max_train_steps:
        args.training_config.max_train_steps = (
            args.training_config.num_train_epochs * num_update_steps_per_epoch
        )
    # Afterwards we recalculate our number of training epochs
    args.training_config.num_train_epochs = math.ceil(
        args.training_config.max_train_steps / num_update_steps_per_epoch
    )

    # Train!
    total_batch_size = (
        args.training_config.train_batch_size
        * dp_size
        * args.training_config.gradient_accumulation_steps
    )
    num_trainable_parameters = sum(
        param.numel() for model in params_to_optimize for param in model["params"]
    )

    print("***** Running training *****")
    print(f"  Num trainable parameters = {num_trainable_parameters}")
    print(f"  Num batches each epoch = {len(train_dataloader)}")
    print(f"  Num epochs = {args.training_config.num_train_epochs}")
    print(
        f"  Instantaneous batch size per device = {args.training_config.train_batch_size}"
    )
    print(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}"
    )
    print(
        f"  Gradient accumulation steps = {args.training_config.gradient_accumulation_steps}"
    )
    print(f"  Total optimization steps = {args.training_config.max_train_steps}")
    first_epoch = 0
    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.training_config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not local_rank == 0,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def calculate_shift(
        image_seq_len,
        base_seq_len: int = 256,
        max_seq_len: int = 4096,
        base_shift: float = 0.5,
        max_shift: float = 1.16,
    ):
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        b = base_shift - m * base_seq_len
        mu = image_seq_len * m + b
        return mu

    def apply_schedule_shift(
        sigmas,
        noise,
        base_seq_len: Optional[int] = None,
        max_seq_len: Optional[int] = None,
        base_shift: Optional[float] = None,
        max_shift: Optional[float] = None,
    ):
        # Resolution-dependent shifting of timestep schedules as per section 5.3.2 of SD3 paper
        # Resolution-dependent shift value calculation used by official Flux inference implementation
        image_seq_len = (
            noise.shape[-1] * noise.shape[-2] * noise.shape[-3]
        ) // 4  # patch size 1,2,2
        mu = calculate_shift(
            image_seq_len,
            base_seq_len or noise_scheduler_copy.config.base_image_seq_len,
            max_seq_len or noise_scheduler_copy.config.max_image_seq_len,
            base_shift or noise_scheduler_copy.config.base_shift,
            max_shift or noise_scheduler_copy.config.max_shift,
        )
        shift = math.exp(mu)
        sigmas = (sigmas * shift) / (1 + (shift - 1) * sigmas)
        return sigmas

    for epoch in range(first_epoch, args.training_config.num_train_epochs):
        train_dataloader.sampler.set_epoch(epoch)
        transformer.train()
        # Track accumulated gradients
        accumulated_batches = 0
        for step, batch in enumerate(train_dataloader):
            # Zero gradients at the beginning of accumulation cycle
            if accumulated_batches == 0:
                optimizer.zero_grad()

            with torch.no_grad():
                # Forward & backward
                # Get data samplex
                videos = batch["videos"]
                videos = videos.to(device, dtype=vae.dtype)
                videos = vae.encode(videos)
                # Get latents (z_0)
                model_input = videos.to(
                    memory_format=torch.contiguous_format, dtype=weight_dtype
                )
                prompts = batch["prompts"]

                # encode prompts
                prompt_embeds = encode_prompt(
                    tokenizer=tokenizer,
                    text_encoder=text_encoder,
                    prompt=prompts,
                    device=device,
                    dtype=weight_dtype,
                )

            # Sample noise that we'll add to the latents
            # Get noise
            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=args.training_config.weighting_scheme,
                batch_size=bsz,
                logit_mean=args.training_config.logit_mean,
                logit_std=args.training_config.logit_std,
                mode_scale=args.training_config.mode_scale,
            )
            indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
            timesteps = noise_scheduler_copy.timesteps[indices].to(
                device=model_input.device
            )
            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=torch.float32)

            if args.model_config.use_dynamic_shifting:
                sigmas = apply_schedule_shift(
                    sigmas,
                    noise,
                    base_seq_len=args.training_config.base_seq_len,
                    max_seq_len=args.training_config.max_seq_len,
                    base_shift=args.training_config.base_shift,
                    max_shift=args.training_config.max_shift,
                )
                timesteps = sigmas * 1000.0  # rescale to [0, 1000.0)

            while sigmas.ndim < model_input.ndim:
                sigmas = sigmas.unsqueeze(-1)

            sigmas = sigmas.to(device=model_input.device, dtype=model_input.dtype)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

            # Predict the noise residual
            model_pred = transformer(
                hidden_states=noisy_model_input,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                is_flash_attn=args.training_config.is_flash_attn,
                return_dict=False,
            )[0]

            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.training_config.weighting_scheme, sigmas=sigmas
            )
            target = noise - model_input

            # Compute regular loss.
            flow_loss = torch.mean(
                (
                    weighting.float() * (model_pred.float() - target.float()) ** 2
                ).reshape(target.shape[0], -1),
                1,
            )
            loss = flow_loss / args.training_config.gradient_accumulation_step

            # Perform backward pass and accumulate gradients
            scaler.scale(loss).backward()

            # Increment accumulation counter
            accumulated_batches += 1

            # If we've accumulated enough gradients, update weights
            if accumulated_batches >= args.training_config.gradient_accumulation_steps:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    transformer_lora_parameters, args.training_config.max_grad_norm
                )
                scaler.step(optimizer)
                optimizer.zero_grad(set_to_none=True)
                scaler.update()
                lr_scheduler.step()
                accumulated_batches = (
                    0  # Reset gradient accumulation counter after the update
                )
                progress_bar.update(1)
                global_step += 1

            # Log to tracker
            if global_rank == 0:
                logs = {
                    "flow_loss": flow_loss.detach().item(),
                    "lr": lr_scheduler.get_last_lr()[0],
                    "grad_norm": grad_norm,
                }
                wandb.log(logs)
                progress_bar.set_postfix(**logs)

            if global_step % args.training_config.checkpointing_steps == 0:
                dist.barrier()
                checkpoint_path = os.path.join(
                    args.output_dir, f"checkpoint-{global_step}"
                )
                save_state(
                    output_dir=checkpoint_path,
                    global_step=global_step,
                    model=unwrap_model(transformer),
                    is_fsdp=args.model_config.fsdp_transformer,
                    optimizer=optimizer,
                    dataloader=train_dataloader,
                    sampler=sampler,
                    scaler=scaler,
                    lr_scheduler=lr_scheduler,
                )

            # Free memory
            del model_pred
            del videos
            del prompt_embeds
            del model_input
            del noise
            del loss
            free_memory()

            if (
                global_step % args.validation_config.validation_steps == 0
                or global_step == 1
            ):
                print(f"Validation {global_rank}")
                pipe = WanPipeline.from_pretrained(
                    args.model_config.pretrained_model_name_or_path,
                    transformer=unwrap_model(transformer),
                    vae=vae,
                    text_encoder=text_encoder,
                    scheduler=noise_scheduler,
                    torch_dtype=weight_dtype,
                )
                validation_prompts = args.validation_config.validation_prompt.split(
                    args.validation_config.validation_prompt_separator
                )

                for validation_prompt in validation_prompts:
                    pipeline_args = {
                        "prompt": validation_prompt,
                        "negative_prompt": "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards",
                        "guidance_scale": args.validation_config.guidance_scale,
                        "num_frames": args.data_config.max_num_frames,
                        "height": args.data_config.height,
                        "width": args.data_config.width,
                        "num_inference_steps": args.validation_config.num_inference_steps,
                        "is_flash_attn": args.training_config.is_flash_attn,
                    }
                    with torch.no_grad() and torch.amp.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        log_validation(
                            pipe=pipe,
                            args=args,
                            pipeline_args=pipeline_args,
                            global_step=global_step,
                            phase_name="validation",
                            global_rank=global_rank,
                        )

            if global_step >= args.training_config.max_train_steps:
                break

    dist.barrier()
    cleanup_distributed_env()


@torch.inference_mode()
def log_validation(
    pipe,
    args: Args,
    pipeline_args,
    global_step,
    phase_name="",
    global_rank=0,
):
    print(
        f"Running validation... \n Generating {args.validation_config.num_validation_videos} videos with prompt: {pipeline_args['prompt']}."
    )

    generator = (
        torch.Generator(device="cuda").manual_seed(args.seed) if args.seed else None
    )

    videos = []
    for _ in range(args.validation_config.num_validation_videos):
        pt_images = pipe(**pipeline_args, generator=generator, output_type="pt").frames[
            0
        ]
        pt_images = torch.stack([pt_images[i] for i in range(pt_images.shape[0])])

        image_np = VaeImageProcessor.pt_to_numpy(pt_images)
        image_pil = VaeImageProcessor.numpy_to_pil(image_np)

        videos.append(image_pil)

    video_filenames = []
    for i, video in enumerate(videos):
        prompt = (
            pipeline_args["prompt"][:25]
            .replace(" ", "_")
            .replace(" ", "_")
            .replace("'", "_")
            .replace('"', "_")
            .replace("/", "_")
        )
        filename = os.path.join(
            args.output_dir,
            f"global_step{global_step}_{phase_name.replace('/', '_')}_video_{i}_{prompt}.mp4",
        )
        if global_rank == 0:
            export_to_video(video, filename, fps=8)
        video_filenames.append(filename)
    if global_rank == 0:
        wandb.log(
            {
                phase_name: [
                    wandb.Video(filename, caption=f"{i}: {pipeline_args['prompt']}")
                    for i, filename in enumerate(video_filenames)
                ]
            }
        )

    del pipe
    free_memory()

    return videos


if __name__ == "__main__":
    from omegaconf import OmegaConf

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    schema = OmegaConf.structured(Args)
    conf = OmegaConf.merge(schema, config)
    main(conf)
