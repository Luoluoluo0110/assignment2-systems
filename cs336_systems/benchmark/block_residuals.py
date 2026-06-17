"""(f) 测量单个 xl TransformerBlock 为反向保存的激活(residuals),按产生它的 op 归类，
并计算该 block 的梯度张量大小。

方法同 PDF §3.2：用 torch.autograd.graph.saved_tensors_hooks 在前向时拦截每个被保存的张量，
记录其字节数与 grad_fn（产生它的算子），最后汇总 + 列出 5 大贡献 op 及占比。
"""
import torch
from collections import defaultdict
from cs336_basics.model import TransformerBlock, RotaryEmbedding

torch.manual_seed(0)
device = "cuda"
# xl 配置
d_model, d_ff, num_heads, context_length = 2560, 10240, 32, 2048
batch = 1  # 与 2.1.6 的 profiling 一致

block = TransformerBlock(
    d_model=d_model, num_heads=num_heads, d_ff=d_ff,
    positional_encoder=RotaryEmbedding(context_length=context_length, dim=d_model // num_heads),
).to(device)

x = torch.randn(batch, context_length, d_model, device=device, requires_grad=True)

MiB = 1024 ** 2
by_op = defaultdict(lambda: [0, 0])  # op_name -> [bytes, count]
total = 0


def pack_hook(t):
    global total
    if isinstance(t, torch.nn.Parameter):   # 跳过参数，避免与权重重复计
        return t
    nbytes = t.numel() * t.element_size()
    op = type(t.grad_fn).__name__ if t.grad_fn is not None else "leaf/input"
    by_op[op][0] += nbytes
    by_op[op][1] += 1
    total += nbytes
    return t


def unpack_hook(t):
    return t


with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = block(x)            # 前向时即触发保存
torch.cuda.synchronize()

print(f"=== single xl TransformerBlock | batch={batch} seq={context_length} d_model={d_model} ===")
print(f"Total residuals saved for backward: {total/MiB:.1f} MiB\n")
print(f"{'producing op (grad_fn)':32s}{'MiB':>10s}{'%':>8s}{'count':>8s}")
for op, (b, c) in sorted(by_op.items(), key=lambda kv: -kv[1][0]):
    print(f"{op:32s}{b/MiB:10.1f}{100*b/total:8.1f}{c:8d}")

# 梯度张量大小 = 该 block 的参数量 × 4 字节（每个参数一个梯度）
nparam = sum(p.numel() for p in block.parameters())
print(f"\nBlock params = {nparam:,} = {nparam*4/MiB:.1f} MiB (fp32)")
print(f"=> gradient tensors produced for one block ~ {nparam*4/MiB:.1f} MiB (= params x 4B)")
