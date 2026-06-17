"""(f) nsys 显存+NVTX profiling 目标脚本：xl 完整训练步，每个 TransformerBlock 打 NVTX 标签。

配合：
  nsys profile --trace=cuda,nvtx --cuda-memory-usage=true -o block_mem \
      .venv/bin/python cs336_systems/benchmark/nsys_mem_target.py
在 Nsight Systems GUI 里：看 CUDA memory 行随每个 block 的变化（forward 爬升 / backward 锯齿）。
"""
import torch
import torch.cuda.nvtx as nvtx
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy

device = "cuda"
cfg = dict(vocab_size=32000, context_length=512, d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
model = BasicsTransformerLM(**cfg).to(device)
model.train()
opt = AdamW(model.parameters())

# 给每个 TransformerBlock 加 NVTX 范围（前向）
# 注意：hook 必须返回 None，否则 pre-hook 的返回值会被当作"替换后的输入"
def _make_pre(i):
    def hook(m, inp):
        nvtx.range_push(f"block{i}")
    return hook

def _post(m, inp, out):
    nvtx.range_pop()

for i, blk in enumerate(model.layers):
    blk.register_forward_pre_hook(_make_pre(i))
    blk.register_forward_hook(_post)

x = torch.randint(0, cfg["vocab_size"], (1, cfg["context_length"]), device=device)
inp, tgt = x[:, :-1], x[:, 1:]


def step():
    opt.zero_grad()
    with nvtx.range("forward"):
        logits = model(inp)
        torch.cuda.synchronize()
    with nvtx.range("backward"):
        loss = cross_entropy(logits.reshape(-1, cfg["vocab_size"]), tgt.reshape(-1))
        loss.backward()
        torch.cuda.synchronize()
    with nvtx.range("optimizer"):
        opt.step()
        torch.cuda.synchronize()


for _ in range(2):   # warmup
    step()

# 被 profile 的那一步，emit_nvtx 让每个 autograd op 也带标签
with torch.autograd.profiler.emit_nvtx():
    step()
torch.cuda.synchronize()
print("done")
