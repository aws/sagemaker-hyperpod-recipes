import torch
import torch.distributed as dist

print("init process group")
dist.init_process_group("nccl")
print("rank:", dist.get_rank())
torch.cuda.set_device(dist.get_rank() % 8)
tensor = torch.randn(4, 4, device="cuda")
print(f"[{dist.get_rank()}] tensor {tensor}")
dist.all_reduce(tensor)
print(f"[{dist.get_rank()}] tensor {tensor} after reduce")
