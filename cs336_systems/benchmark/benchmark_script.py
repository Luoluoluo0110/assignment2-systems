import argparse
import timeit
import statistics
import torch
import torch.nn as nn

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
    
    # 遍历model_configs, 如果这个字典的name属性跟model_name一样, next() 作用就是从迭代对象里拿第一个元素 
    select_cfg = next(cfg for cfg in model_configs if cfg["name"] == args.model_name)
    # 初始化模型 **select_cfg: 字典解包语法，把字典的键值对批量转化成关键词参数 
    # *xxx：解包列表 / 元组，用于位置参数
    # **xxx：解包字典，用于关键字参数
    model_cfg = {k: v for k, v in select_cfg.items() if k != "name"}
    model = BasicsTransformerLM(**model_cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ====================== 实现下面这几块 ======================
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
        if args.mode == "forward":
            with torch.no_grad():
        # logits.shape [batch_size, seq_len, vocab_size]
                logits = model(input_seq)
        else:
            # 反向/训练模式：保留计算图，正常求导
            logits = model(input_seq)

        # 但是cross_entropy 传入的是logits是(batch_size * seq_len, vocab_size), target(batch_size * seq_len)
        if args.mode in ("forward_backward", "full_train"):            
            loss = cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            loss.backward()
        if args.mode == "full_train":
            optimizer.step()

        # 【必加】GPU 同步，固定位置
        torch.cuda.synchronize()

    # 8. 正式测速阶段
    print(f"Running {args.test_steps} benchmark steps ...")
    time_records = []  # 存每一轮耗时

    for _ in range(args.test_steps):
        # 开始计时
        start = timeit.default_timer()
        if optimizer is not None:
            optimizer.zero_grad()
        # 纯推理：进入 no_grad 上下文，关闭计算图
        input_seq = dummy_input[:, :-1]
        target_seq = dummy_input[:, 1:]
        if args.mode == "forward":
            with torch.no_grad():
                logits = model(input_seq)
        else:
            # 反向/训练模式：保留计算图，正常求导
            logits = model(input_seq)
        if args.mode in ("forward_backward", "full_train"):            
            loss = cross_entropy(logits.reshape(-1, vocab_size), target_seq.reshape(-1))
            loss.backward()
        if args.mode == "full_train":
            optimizer.step()

        torch.cuda.synchronize()

        # 结束计时，记录耗时
        end = timeit.default_timer()
        time_records.append(end - start)

    # 统计均值、标准差
    avg_time = statistics.mean(time_records)
    std_time = statistics.stdev(time_records)

    print(f"Average time per step: {avg_time:.4f} s")
    print(f"Standard deviation: {std_time:.4f} s")


if __name__ == "__main__":
    main()