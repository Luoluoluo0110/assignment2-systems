import argparse
import timeit
import statistics
import torch
import torch.nn as nn
import os

from contextlib import nullcontext
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.nn_utils import cross_entropy
def main():
    parser = argparse.ArgumentParser(description="Transformer End-to-End Benchmark")
    parser.add_argument(
        "--model_name",
        type=str,
        choices=["small", "medium", "large", "xl", "10B"],
        required=True,
        help="Select model size"
        )

    # 运行模式：区分 forward / forward+backward / full_train
    parser.add_argument(
        "--mode",
        type=str,
        choices=["forward", "forward_backward", "full_train"],
        default="forward"
    )

    # 步数配置
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--test_steps", type=int, default=10)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--bf16", action="store_true", default=None)
    parser.add_argument("--memory_profile", action="store_true", default=None)
    parser.add_argument("--profile_step", type=int, default=3)
    parser.add_argument("--snapshot_dir", type=str, default='.')
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model_configs = [
        {
            "name": "small",
            "vocab_size": 32000,
            "context_length": 512,
            "d_model": 768,
            "d_ff": 3072,
            "num_layers": 12,
            "num_heads": 12
        },
        {
            "name": "medium",
            "vocab_size": 32000,
            "context_length": 512,
            "d_model": 1024,
            "d_ff": 4096,
            "num_layers": 24,
            "num_heads": 16
        },
        {
            "name": "large",
            "vocab_size": 32000,
            "context_length": 512,
            "d_model": 1280,
            "d_ff": 5120,
            "num_layers": 36,
            "num_heads": 20
        },
        {
            "name": "xl",
            "vocab_size": 32000,
            "context_length": 512,
            "d_model": 2560,
            "d_ff": 10240,
            "num_layers": 32,
            "num_heads": 32
        },
        {
            "name": "10B",
            "vocab_size": 32000,
            "context_length": 512,
            "d_model": 4608,
            "d_ff": 12288,
            "num_layers": 50,
            "num_heads": 36
        }
    ]

    # 解析命令行的参数，将参数值打包成一个对象
    args = parser.parse_args()
    precision = "bf16" if args.bf16 else "fp32"
    ctx = torch.autocast(device_type=device, dtype=torch.bfloat16) if args.bf16 else nullcontext()
    

    
    # 遍历model_configs, 如果这个字典的name属性跟model_name一样, next() 作用就是从迭代对象里拿第一个元素 
    select_cfg = next(cfg for cfg in model_configs if cfg["name"] == args.model_name)
    # 初始化模型 **select_cfg: 字典解包语法，把字典的键值对批量转化成关键词参数 
    # *xxx：解包列表 / 元组，用于位置参数
    # **xxx：解包字典，用于关键字参数
    if args.context_length is not None:
        select_cfg["context_length"] = args.context_length

    model_cfg = {k: v for k, v in select_cfg.items() if k != "name"}
    model = BasicsTransformerLM(**model_cfg)

    # 4. 移动到设备
    model.to(device)
    if args.mode == "forward":
        model.eval()
    else:
        model.train()
    
    # 5. 初始化优化器 (full_train 模式才需要)
    optimizer = None
    if args.mode in ("forward_backward", "full_train"):
        optimizer = AdamW(model.parameters())

    # 6. 生成随机输入数据 + 移动到设备
    batch_size = 1
    seq_len = select_cfg["context_length"]
    vocab_size = select_cfg["vocab_size"]
    # torch.randint语法: 生成一个(最小值，最大值，形状)张量
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    dummy_input = dummy_input.to(device)
    # =================================================================

    # 7. 热身阶段：只运行，不计时
    print(f"Running {args.warmup_steps} warmup steps ...")
    for _ in range(args.warmup_steps):
        # ---------------- 你补单轮执行逻辑 ----------------
        # 根据 mode 写 forward / backward / optimizer.step
        # -------------------------------------------------
        if optimizer is not None:
            optimizer.zero_grad()
        input_seq = dummy_input[:, :-1]
        target_seq = dummy_input[:, 1:]
        # 纯推理：进入 no_grad 上下文，关闭计算图:
        with ctx:
            if args.mode == "forward":
                with torch.no_grad():
                    # logits.shape [batch_size, seq_len, vocab_size]
                    logits = model(input_seq)
            else:
                # 反向/训练模式：保留计算图，正常求导
                logits = model(input_seq)

        # 但是cross_entropy 传入的是logits是(batch_size * seq_len, vocab_size), target(batch_size * seq_len)
        if args.mode in ("forward_backward", "full_train"):
            with ctx:            
                loss = cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            loss.backward()
        if args.mode == "full_train":
            optimizer.step()

        # 【必加】GPU 同步，固定位置
        torch.cuda.synchronize()

    # 分析显存
    # pytorch 维护一个caching allocator.张量释放，池子里复用
    if args.memory_profile:
        # 把池子里空闲的缓存还给 GPU,让起点干净。(注意:它只释放空闲的,正在用的不动。)
        # allocated:真正被张量占用的量
        # reserved:PyTorch 向 GPU 要来的池子总量(≥ allocated)

        torch.cuda.empty_cache()
        # 把"峰值"计数器清零。
        torch.cuda.reset_peak_memory_stats()
        # 打开显存时间记录器，最多记100万条事件 
        torch.cuda.memory._record_memory_history(max_entries=1000000)
        for _ in range(args.profile_step):
            if optimizer is not None:
                optimizer.zero_grad()
            input_seq = dummy_input[:, :-1]
            target_seq = dummy_input[:, 1:]
        
            with ctx:
                if args.mode == "forward":
                    with torch.no_grad():
                        logits = model(input_seq)
                else:
                    logits = model(input_seq)
            # 算loss
            if args.mode in ("forward_backward", "full_train"):
                with ctx:
                    loss = cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
                loss.backward()
            # 更新梯度
            if args.mode == "full_train":
                optimizer.step()
        torch.cuda.synchronize()
        cl = select_cfg["context_length"]
        fname = f"memsnap_{args.model_name}_ctx{cl}_{args.mode}_{precision}.pickle"
        os.makedirs(args.snapshot_dir, exist_ok=True)
        path = os.path.join(args.snapshot_dir, fname)
        torch.cuda.memory._dump_snapshot(path)
        torch.cuda.memory._record_memory_history(enabled=None)
        peak = torch.cuda.max_memory_allocated() / 1024**2               
        print(f"[MEM {args.model_name} ctx{cl} {args.mode} {precision}] peak={peak:.1f} MiB")
        return

    # 正式测速阶段
    print(f"Running {args.test_steps} benchmark steps ...")
    time_records = []   # 整步耗时
    fwd_records = []    # 仅 forward 耗时
    bwd_records = []    # 仅 backward 耗时

    for _ in range(args.test_steps):
        if optimizer is not None:
            optimizer.zero_grad()
        input_seq = dummy_input[:, :-1]
        target_seq = dummy_input[:, 1:]

        # 整步开始
        step_start = timeit.default_timer()

        # ---------------- forward（单独计时）----------------
        # synchronize 放在区间内部，撑到 GPU kernel 执行完再关闭计时
        fwd_start = timeit.default_timer()
        with torch.cuda.nvtx.range("forward"):
            with ctx:
                if args.mode == "forward":
                    with torch.no_grad():
                        logits = model(input_seq)
                else:
                    # 反向/训练模式：保留计算图，正常求导
                    logits = model(input_seq)
            torch.cuda.synchronize()
        fwd_records.append(timeit.default_timer() - fwd_start)

        # ---------------- backward（单独计时）----------------
        if args.mode in ("forward_backward", "full_train"):
            # loss 计算不计入 backward 计时
            with torch.cuda.nvtx.range("loss"):
                with ctx:
                    loss = cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
                torch.cuda.synchronize()
            bwd_start = timeit.default_timer()
            with torch.cuda.nvtx.range("backward"):
                loss.backward()
                torch.cuda.synchronize()
            bwd_records.append(timeit.default_timer() - bwd_start)

        if args.mode == "full_train":
            with torch.cuda.nvtx.range("optimizer.step"):
                optimizer.step()
                torch.cuda.synchronize()

        # 整步结束（最后一个区间已 synchronize）
        time_records.append(timeit.default_timer() - step_start)

    # 统计均值、标准差
    def _stats(xs):
        if not xs:
            return float("nan"), float("nan")
        return statistics.mean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)

    fwd_mean, fwd_std = _stats(fwd_records)
    bwd_mean, bwd_std = _stats(bwd_records)
    tot_mean, tot_std = _stats(time_records)

    print(f"[{args.model_name} | {precision} | {args.mode}] "
          f"forward {fwd_mean:.4f}±{fwd_std:.4f}s | "
          f"backward {bwd_mean:.4f}±{bwd_std:.4f}s | "
          f"total {tot_mean:.4f}±{tot_std:.4f}s")
    # 机器可解析的汇总行，供 sweep 脚本收集
    print(f"RESULT model={args.model_name} precision={precision} mode={args.mode} "
          f"fwd_mean={fwd_mean:.6f} fwd_std={fwd_std:.6f} "
          f"bwd_mean={bwd_mean:.6f} bwd_std={bwd_std:.6f} "
          f"tot_mean={tot_mean:.6f} tot_std={tot_std:.6f}")


if __name__ == "__main__":
    main()