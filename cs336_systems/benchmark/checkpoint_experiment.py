"""§3 (b) 梯度检查点峰值显存实验。

把 xl 的 32 个 TransformerBlock 按 group_size 分组，每组用 torch.utils.checkpoint 包起来
（单层，不嵌套），测 forward+backward 的峰值显存。group_size=0 表示不做 checkpoint（基线）。

用法:
    python checkpoint_experiment.py --group_size 4        # 每 4 层一个 checkpoint
    python checkpoint_experiment.py --group_size 0        # 基线（无 checkpoint）
"""
import argparse
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


class CkptStack(nn.Module):
    """把一串 block 按 group_size 分组，每组一次 checkpoint（enabled=False 时退化为普通顺序执行）。"""
    def __init__(self, blocks, group_size, enabled):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)
        self.gs = max(1, group_size)
        self.enabled = enabled

    def forward(self, x):
        n = len(self.blocks)
        for i in range(0, n, self.gs):
            grp = self.blocks[i:i + self.gs]

            def run(x, grp=grp):
                for b in grp:
                    x = b(x)
                return x

            if self.enabled:
                x = checkpoint(run, x, use_reentrant=False)
            else:
                x = run(x)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group_size", type=int, default=0, help="0=no checkpoint; >0 = blocks per checkpoint")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    dev = "cuda"
    vocab = 32000
    cfg = dict(vocab_size=vocab, context_length=args.seq, d_model=2560, d_ff=10240, num_layers=32, num_heads=32)
    model = BasicsTransformerLM(**cfg).to(dev)
    model.train()

    enabled = args.group_size > 0
    orig = list(model.layers)
    model.layers = nn.ModuleList([CkptStack(orig, args.group_size, enabled)]).to(dev)

    x = torch.randint(0, vocab, (args.batch, args.seq), device=dev)
    inp, tgt = x[:, :-1], x[:, 1:]

    def one_step():
        loss = cross_entropy(model(inp).reshape(-1, vocab), tgt.reshape(-1))
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(2):          # warmup
        one_step()
        torch.cuda.synchronize()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        one_step()
    torch.cuda.synchronize()

    peak = torch.cuda.max_memory_allocated() / 1024 ** 2
    tag = f"g{args.group_size}" if enabled else "none"
    print(f"[CKPT {tag} batch{args.batch} seq{args.seq}] peak={peak:.1f} MiB")


if __name__ == "__main__":
    main()