import sys

sys.path.append(".")
from spark_wan.models.transformer_wan import WanTransformer3DModel, WanTransformerBlock
from spark_wan.training_utils.gan_utils import calculate_adaptive_weight
from peft import get_peft_model, LoraConfig
import torch
from spark_wan.training_utils.fsdp2_utils import prepare_fsdp_model
from spark_wan.training_utils.load_model import replace_rmsnorm_with_fp32, load_model

weight_dtype = torch.float16

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

class DummyDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(5*16*16*16, 1),
        )
        
    def forward(self, x):
        return self.net(x)
    
discriminator = DummyDiscriminator()
discriminator = discriminator.to("cuda", dtype=weight_dtype)

transformer.disable_adapter_layers()
input_hidden_states = torch.randn(1, 16, 5, 16, 16, device="cuda", dtype=weight_dtype)
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
	hidden_states=torch.randn(1, 16, 5, 16, 16, device="cuda", dtype=weight_dtype),
	timestep=torch.tensor([1000], device="cuda", dtype=torch.long),
	encoder_hidden_states=torch.randn(1, 512, 4096, device="cuda", dtype=weight_dtype),
	return_dict=False
)[0]

loss = torch.sum(torch.abs(student_output - teacher_output))
score = torch.sum(discriminator(student_output.flatten(start_dim=1)))
print(torch.autograd.grad(loss, transformer.base_model.blocks[-1].ffn.net[2].lora_A.default.weight, retain_graph=True))
