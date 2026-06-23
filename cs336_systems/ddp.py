import torch
import torch.nn as nn
import torch.distributed as dist


# 初始化：
#   rank 0 的模型参数 → broadcast → 所有 rank（保证起点一致）

# 每一个训练步：
#   1. 每个 rank 拿到自己那份数据（batch 的一个切片）
#   2. 各自 forward → 各自算 loss → 各自 backward（得到本地梯度）
#   3. 【关键】all-reduce 梯度（所有 rank 的梯度求平均）
#   4. 每个 rank 用同样的平均梯度 optimizer.step()
#      → 因为参数起点一样、梯度一样，更新后参数也完全一样
#   5. 循环

class DDP(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
        self.handles = []
        seen = set()
        for p in module.parameters():
            if id(p) in seen:
                continue
            seen.add(id(p))
            dist.broadcast(p.data, src = 0)
            # hook
            if p.requires_grad:
                p.register_post_accumulate_grad_hook(lambda p: 
                self.handles.append(dist.all_reduce(p.grad, async_op=True)))

        for b in module.buffers():
            dist.broadcast(b.data, src = 0)
        
        
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def finish_gradient_synchronization(self):
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        for p in self.module.parameters():
            if p.requires_grad and p.grad is not None:
                p.grad /=  dist.get_world_size()