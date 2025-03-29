import sys
sys.path.append(".")
import torch
from spark_wan.training_utils.fsdp2_utils import prepare_fsdp_model

torch.distributed.init_process_group(backend="nccl")
torch.cuda.set_device(torch.distributed.get_rank())

class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

model = DummyModel()
prepare_fsdp_model(
    model,
    shard_conditions=[lambda name, module: "linear" in name],
    reshard_after_forward=False,
    cpu_offload=False,
    weight_dtype=torch.bfloat16,
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
scaler = torch.amp.GradScaler(enabled=True)

x = torch.randn(1, 10, device="cuda").to(torch.bfloat16)

for i in range(1000):
    y = model(x)
    loss = y.mean()
    if torch.distributed.get_rank() == 0:
        print(loss)
    scaler.scale(loss).backward()
    print(model.linear.weight.grad)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    print(model.linear.weight.grad)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()