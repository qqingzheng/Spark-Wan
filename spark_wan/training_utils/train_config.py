
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ReportTo:
    wandb: bool = field(default=True)
    project_name: str = field(default="Spark-Wan")
    wandb_name: str = field(default="testrun")
    wandb_notes: str = field(default="")

@dataclass
class DataConfig:
    instance_data_root: str = field(default="", metadata={"required": True})
    height: int = field(default=480)
    width: int = field(default=720)
    fps: int = field(default=8)
    max_num_frames: int = field(default=49)
    dataloader_num_workers: int = field(default=0)

@dataclass
class ModelConfig:
    pretrained_lora_path: Optional[str] = field(default=None)
    pretrained_model_name_or_path: Optional[str] = field(
        default=None,
    )
    compile_transformer: bool = field(default=False)
    fsdp_transformer: bool = field(default=False)
    fsdp_text_encoder: bool = field(default=False)
    sp_size: int = field(default=1)
    is_train_lora: bool = field(default=False)
    lora_rank: int = field(default=128)
    lora_alpha: float = field(default=128)
    lora_dropout: float = field(default=0.0)
    lora_target_modules: Optional[str] = field(default=None)
    flow_shift: float = field(default=8.0)
    
@dataclass
class ValidationConfig:
    validation_steps: int = field(default=100)
    validation_prompt: Optional[str] = field(default="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse. She wears sunglasses and red lipstick. She walks confidently and casually. The street is damp and reflective, creating a mirror effect of the colorful lights. Many pedestrians walk about.")
    validation_prompt_separator: str = field(default=":::")
    num_validation_videos: int = field(default=1)

@dataclass
class TrainingConfig:
    resume_from_checkpoint: Optional[str] = field(default=None)
    gradient_accumulation_steps: int = field(default=1)
    max_train_steps: Optional[int] = field(default=None)
    checkpointing_steps: int = field(default=500)
    train_batch_size: int = field(default=4)
    num_train_epochs: int = field(default=1)
    gradient_checkpointing: bool = field(default=False)
    mixed_precision: str = field(
        default="bf16",
        metadata={"choices": ["no", "fp16", "bf16"]},
    )
    learning_rate: float = field(default=1e-4)
    lr_scheduler: str = field(
        default="constant",
        metadata={
            "choices": [
                "linear",
                "cosine",
                "cosine_with_restarts",
                "polynomial",
                "constant",
                "constant_with_warmup",
            ]
        },
    )
    lr_warmup_steps: int = field(default=500)
    lr_num_cycles: int = field(default=1)
    lr_power: float = field(default=1.0)
    optimizer: str = field(
        default="adam",
        metadata={
            "choices": ["adam", "adamw", "prodigy"],
        },
    )
    adam_beta1: float = field(default=0.9)
    adam_beta2: float = field(default=0.95)
    adam_weight_decay: float = field(default=1e-04)
    adam_epsilon: float = field(default=1e-08)
    max_grad_norm: float = field(default=1.0)

@dataclass
class ParallelConfig:
    sp_size: int = field(default=1)

@dataclass
class StepDistillConfig:
    teacher_step: int = field(default=32)
    student_step: int = field(default=16)
    guidance_scale_max: float = field(default=8.0)
    guidance_scale_min: float = field(default=4.0)
    is_gan_distill: bool = field(default=False)
    disc_interval: int = field(default=1)
    disc_start: int = field(default=0)
    distance_weight: float = field(default=1.0)
    disc_weight: float = field(default=1.0)
    
@dataclass
class Args:
    output_dir: str = field(default="wan-lora")
    seed: int = field(default=1234)
    report_to: ReportTo = field(default_factory=ReportTo)
    parallel_config: ParallelConfig = field(default_factory=ParallelConfig)
    data_config: DataConfig = field(default_factory=DataConfig)
    model_config: ModelConfig = field(default_factory=ModelConfig)
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    step_distill_config: Optional[StepDistillConfig] = field(default=None)
    logging_dir: str = field(default="logs")