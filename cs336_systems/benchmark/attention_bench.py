"""
Problem (pytorch_attention) 基准脚本 —— 诚实且安全版（针对 WSL ~1.5GB 进程预算）。

按题目要求：batch=8、无多头，遍历 d_model×seq_len 笛卡尔积，对每个配置：
  (iv) 计时 100 次前向；
  (v)  测"反向开始前"的显存占用，并计时 100 次反向；
  (vi) warmup + 每次前/反向后 torch.cuda.synchronize()。
输出一张表：fwd(ms) | 反向前显存(MiB) | bwd(ms)，跑不动的标 OOM / SKIP。

安全结构（见 git 历史里的演进）：
  * 父进程只起 1 个子进程跑完整轮、自己不碰 CUDA → 子进程怎么死都不波及 VSCode/整机；
  * 子进程开头裸分配实测真实可用显存预算（WSL 上 mem_get_info / set_per_process_
    memory_fraction 读数不可信，唯一可信的是实测）；
  * 超预算的配置诚实 SKIP，不去触发必败分配；够得着的才真跑。

省显存旋钮（题目固定 batch=8；这些用于自己探索更大 seq）：
  --batch N / --dtype fp32|bf16|fp16 / --fwd-only / --iters N
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

HIDDEN_DIM = [16, 32, 64, 128]
SEQ_LEN = [256, 1024, 4096, 8192, 16384]
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}

# 峰值显存经验系数：peak ≈ K_PEAK × 分数矩阵大小（分数+softmax+两者的梯度≈4~5 份）。
# 取 5（fwd+bwd）/ 3（fwd-only）；够不着的会干净 OOM，安全，所以系数取实而非过度保守。
K_PEAK_BWD = 5
K_PEAK_FWD = 3


def parse_args(argv) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="pytorch_attention 基准（WSL 安全版）")
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--dtype", choices=list(DTYPE_BYTES), default="fp32")
    p.add_argument("--fwd-only", action="store_true")
    p.add_argument("--iters", type=int, default=100, help="计时迭代次数（题目要求 100）")
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def score_matrix_mib(batch: int, seq_len: int, dtype: str) -> float:
    return batch * seq_len * seq_len * DTYPE_BYTES[dtype] / 1024 ** 2


def measure_budget_mib(torch, device) -> int:
    """裸分配实测本进程真实可用显存（MiB）—— WSL 上唯一可信的预算来源。"""
    keep = []
    blk = 64 * 1024 * 1024 // 4
    try:
        while True:
            keep.append(torch.empty(blk, dtype=torch.float32, device=device))
    except torch.cuda.OutOfMemoryError:
        pass
    n = len(keep)
    del keep
    torch.cuda.empty_cache()
    return n * 64


def run_child(args: argparse.Namespace) -> None:
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    import torch
    from cs336_basics.model import scaled_dot_product_attention

    torch_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.dtype]
    do_backward = not args.fwd_only
    k_peak = K_PEAK_BWD if do_backward else K_PEAK_FWD
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == "cuda"

    if not is_cuda:
        print("WARNING: CUDA 不可用，跑在 CPU 上", flush=True)
        budget = 10 ** 9
    else:
        _ = torch.randn(8, device=device) @ torch.randn(8, device=device)   # 预热上下文
        torch.cuda.synchronize()
        budget = measure_budget_mib(torch, device)
        free, total = torch.cuda.mem_get_info()
        print(f"# 实测可用显存预算 ≈ {budget/1024:.2f} GiB"
              f"（显卡物理空闲 {free/1024**3:.2f} / 总 {total/1024**3:.2f} GiB —— WSL/WDDM 压低了进程预算）",
              flush=True)
    mode = "fwd-only" if args.fwd_only else "fwd+bwd"
    print(f"# 配置：batch={args.batch}  dtype={args.dtype}  {mode}  iters={args.iters}", flush=True)

    def sync():
        if is_cuda:
            torch.cuda.synchronize()

    def bench_one(d_model: int, seq_len: int) -> str:
        predicted = k_peak * score_matrix_mib(args.batch, seq_len, args.dtype)
        if is_cuda and predicted > 1.3 * budget:
            return f"SKIP (预测~{predicted/1024:.1f}GiB>预算{budget/1024:.1f}GiB)"
        Q = K = V = out = loss = None
        try:
            shape = (args.batch, seq_len, d_model)
            Q = torch.randn(shape, device=device, dtype=torch_dtype, requires_grad=do_backward)
            K = torch.randn(shape, device=device, dtype=torch_dtype, requires_grad=do_backward)
            V = torch.randn(shape, device=device, dtype=torch_dtype, requires_grad=do_backward)

            # ---- 前向计时（含 warmup）----
            for _ in range(args.warmup):
                out = scaled_dot_product_attention(Q, K, V)
            sync()
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = scaled_dot_product_attention(Q, K, V)
            sync()
            fwd_ms = (time.perf_counter() - t0) / args.iters * 1000

            if not do_backward:
                return f"{fwd_ms:8.3f} |        - |        -"

            # ---- 反向开始前的显存占用 ----
            out = scaled_dot_product_attention(Q, K, V)
            loss = out.float().sum()
            sync()
            mem_before_bwd = torch.cuda.memory_allocated() / 1024 ** 2 if is_cuda else 0.0
            loss.backward()   # 消化掉这次的图

            # ---- 反向计时：每次重建前向(不计时)，只计时 backward ----
            for _ in range(args.warmup):
                o = scaled_dot_product_attention(Q, K, V); o.float().sum().backward()
            sync()
            tot = 0.0
            for _ in range(args.iters):
                o = scaled_dot_product_attention(Q, K, V)
                l = o.float().sum()
                sync()
                tb = time.perf_counter()
                l.backward()
                sync()
                tot += time.perf_counter() - tb
            bwd_ms = tot / args.iters * 1000
            return f"{fwd_ms:8.3f} | {mem_before_bwd:8.1f} | {bwd_ms:8.3f}"
        except torch.cuda.OutOfMemoryError:
            return f"OOM (超{budget/1024:.1f}GiB预算)"
        finally:
            del Q, K, V, out, loss
            if is_cuda:
                torch.cuda.empty_cache()

    print(f"\n{'d_model':>7} {'seq':>6} | {'fwd(ms)':>8} | {'反向前(MiB)':>10} | {'bwd(ms)':>8}", flush=True)
    print("-" * 56, flush=True)
    for d_model, seq_len in itertools.product(HIDDEN_DIM, SEQ_LEN):
        print(f"{d_model:>7} {seq_len:>6} | {bench_one(d_model, seq_len)}", flush=True)


def run_sweep(argv) -> None:
    proc = subprocess.Popen(
        [sys.executable, __file__, "--child", *argv],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if proc.returncode != 0:
        print(f"\n[!] 子进程异常退出 rc={proc.returncode}（已隔离，父进程/VSCode 安全）。"
              f"以上为中断前已测结果。", flush=True)


def main():
    argv = sys.argv[1:]
    args = parse_args(argv)
    if args.child:
        run_child(args)
    else:
        run_sweep(argv)


if __name__ == "__main__":
    main()
