import torch
import torch.nn
import torch.distributed as dist


class ShardedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, optimizer_cls, **kwargs):
        self.optimizer_cls = optimizer_cls
        self.kwargs = kwargs
        self.inner_optimizer = None
        super().__init__(params, defaults={})
    
    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        my_params = param_group['params'][rank::world_size]
        if self.inner_optimizer is None:
            self.inner_optimizer = self.optimizer_cls(my_params, **self.kwargs)
        else:
            self.inner_optimizer.add_param_group({'params': my_params})

    
    def step(self, closure=None, **kwargs):
        self.inner_optimizer.step(closure)
        world_size = dist.get_world_size()
        for group in self.param_groups:
            for i, p in enumerate(group['params']):
                src = i % world_size
                dist.broadcast(p.data, src=src)