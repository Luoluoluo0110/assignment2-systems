# 混合精度 (BF16 autocast) vs 全精度 (FP32) 基准测试

**硬件**: NVIDIA A100-PCIE-40GB, driver 550.90.07 / CUDA 12.4, torch 2.5.1+cu124
**设置**: batch=1, context_length=512, mode=forward_backward, warmup=5, steps=10
**精度**: FP32 = `nullcontext()`；BF16 = `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
（autocast 只把 matmul/conv 等转成 BF16 在 Tensor Core 上跑，权重主副本与归约累加仍为 FP32）

## 结果（时间单位 ms，mean；speedup = FP32 / BF16，>1 表示 BF16 更快）

| 尺寸 | fwd FP32 | fwd BF16 | fwd↑ | bwd FP32 | bwd BF16 | bwd↑ | total FP32 | total BF16 | total↑ |
|---|---|---|---|---|---|---|---|---|---|
| small  | 25.9  | 30.7  | 0.84× | 34.9  | 39.6  | 0.88× | 61.8  | 71.4  | 0.87× |
| medium | 51.6  | 58.5  | 0.88× | 93.5  | 78.2  | 1.20× | 146.9 | 138.4 | 1.06× |
| large  | 106.4 | 86.3  | 1.23× | 200.2 | 116.2 | 1.72× | 310.4 | 205.0 | 1.51× |
| xl     | 252.7 | 79.8  | 3.17× | 526.4 | 160.5 | 3.28× | 782.5 | 243.4 | 3.21× |
| 10B    | OOM   | OOM   | —     | OOM   | OOM   | —     | OOM   | OOM   | —     |

## 趋势分析

1. **加速比随模型变大单调上升**：small 时 BF16 反而更慢（~0.85×），medium 基本持平，large 约 1.5×，
   xl 高达 ~3.2×。原因：小模型每个 kernel 的计算量小、不是 compute-bound，autocast 的类型转换与
   kernel 启动开销占比相对大；而且 A100 对 FP32 matmul 默认走 TF32，本身已较快，削弱了 BF16 的优势。
   模型越大越受 matmul 算力限制，Tensor Core 的 BF16 吞吐优势才充分体现。

2. **backward 比 forward 受益更多**：反向的 matmul 量约为前向两倍（对激活和对权重各一次），
   因此 Tensor Core 加速的绝对收益更大（如 large：fwd 1.23× vs bwd 1.72×）。

3. **混合精度不省权重显存**：10B 在 FP32 与 BF16 下都在 `model.to(device)` 处 OOM——
   autocast 不改变参数的存储 dtype（仍 FP32，~39GB 权重），40GB 卡放不下。
   要真正省显存需用全 BF16 权重 / FSDP / 梯度检查点等手段（后续 part）。

## 复现

```bash
cd assignment2-systems
PYTHONPATH=cs336-basics:. python cs336_systems/benchmark/run_sweep.py
# 或单个组合：
PYTHONPATH=cs336-basics:. python cs336_systems/benchmark/benchmark_script.py \
    --model_name large --mode forward_backward --bf16 --warmup_steps 5 --test_steps 10
```
