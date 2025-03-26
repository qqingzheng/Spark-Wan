import sys

sys.path.append(".")
from spark_wan.models.transformer_wan import WanTransformer3DModel, WanTransformerBlock
from spark_wan.training_utils.gan_utils import calculate_adaptive_weight
from peft import get_peft_model, LoraConfig
import torch
from spark_wan.training_utils.fsdp2_utils import prepare_fsdp_model
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32, load_model

def init_distributed_env():
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)


def cleanup_distributed_env():
    torch.distributed.destroy_process_group()


init_distributed_env()
weight_dtype = torch.float16

# lora_target_modules = [
#         "ffn.net.2",
# ]
# tokenizer, text_encoder, transformer, vae = load_model(
#     pretrained_model_name_or_path="/storage/ysh/Ckpts/Wan2.1-T2V-1.3B-Diffusers/",
#     compile_transformer=False,
#     fsdp_transformer=True,
#     fsdp_text_encoder=False,
#     weight_dtype=weight_dtype,
#     device="cuda",
#     is_train_lora=True,
#     lora_rank=8,
#     lora_alpha=16,
#     lora_dropout=0.1,
#     lora_target_modules=lora_target_modules,
#     pretrained_lora_path=None,
# )
lora_target_modules = [
    "ffn.net.2",
]
transformer = WanTransformer3DModel.from_pretrained(
    "/storage/ysh/Ckpts/Wan2.1-T2V-1.3B-Diffusers/",
    subfolder="transformer",
)
transformer = replace_rmsnorm_with_fp32(transformer)
transformer.enable_gradient_checkpointing()
transformer.requires_grad_(False)
transformer = get_peft_model(transformer, LoraConfig(
    r=8,
    target_modules=lora_target_modules,
    lora_alpha=16,
    lora_dropout=0.1,
    init_lora_weights=True,
))

prepare_fsdp_model(
    transformer,
    shard_conditions=[lambda n, m: isinstance(m, WanTransformerBlock)],
    cpu_offload=False,
    reshard_after_forward=False,
    weight_dtype=weight_dtype,
)

transformer.disable_adapter_layers()
input_hidden_states = torch.randn(1, 16, 21, 80, 80, device="cuda", dtype=weight_dtype)
for i in range(2):
    with torch.no_grad():
        input_hidden_states = transformer(
            hidden_states=input_hidden_states,
            timestep=torch.tensor([1000], device="cuda", dtype=torch.long),
            encoder_hidden_states=torch.randn(1, 512, 4096, device="cuda", dtype=weight_dtype),
            return_dict=False,
        )[0]
teacher_output = input_hidden_states
transformer.enable_adapter_layers()

student_output = transformer(
	hidden_states=torch.randn(1, 16, 21, 80, 80, device="cuda", dtype=weight_dtype),
	timestep=torch.tensor([1000], device="cuda", dtype=torch.long),
	encoder_hidden_states=torch.randn(1, 512, 4096, device="cuda", dtype=weight_dtype),
	return_dict=False
)[0]

loss = torch.sum(torch.abs(student_output - teacher_output))
torch.autograd.grad(loss, transformer.base_model.blocks[-1].ffn.net[2].lora_A.default.weight, retain_graph=True)
# calculate_adaptive_weight(loss, loss, last_layer=[transformer.base_model.blocks[-1].ffn.net[2].lora_A.default.weight])
cleanup_distributed_env()
