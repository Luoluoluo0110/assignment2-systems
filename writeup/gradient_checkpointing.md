# §3 Problem (gradient_checkpointing) — Memory-Optimal Gradient Checkpointing

模型:N 个相同 TransformerBlock 顺序堆叠。设单个 block 为反向保存的激活为 **M**,
checkpoint 在分界处保存的输入激活为 **a**(题设 a ≪ M：residuals dominate bookkeeping)。

## (a) 最小化峰值激活显存的策略（不计算力代价）

**策略:递归嵌套 checkpoint(recursive checkpointing)。** 把 N 个 block 一分为二,
每半再用 checkpoint 包住、内部再一分为二……如此递归。每深入一层嵌套,同时"活着"的
完整 residual 段就少一半,代价是该段在反向时被多重算一次。

| 策略 | 峰值激活显存 | 额外算力 |
|---|---|---|
| 不 checkpoint | O(N) | 0(1× forward) |
| 单层 √N 分组 | O(√N) | O(N)(+1× forward) |
| **递归嵌套(本题答案)** | **O(log N)** | **O(N log N)** |

> 渐进推导:单层分 g 组、每组 s=N/s 个 block,峰值 = O(stored inputs g) + O(recompute one group s)
> = O(N/s + s),取 s=√N 得 O(√N)。递归把"recompute one group"这一项继续二分,
> 峰值降到 O(log N),代价是每层嵌套多一次重算 → 总算力 O(N log N)。

**代码草图**:

```python
from torch.utils.checkpoint import checkpoint

def run_recursive(blocks):
    if len(blocks) == 1:
        return blocks[0]
    mid = len(blocks) // 2
    left, right = run_recursive(blocks[:mid]), run_recursive(blocks[mid:])
    def fn(x):
        x = checkpoint(left,  x, use_reentrant=False)   # 嵌套：内部还会再 checkpoint
        x = checkpoint(right, x, use_reentrant=False)
        return x
    return fn
```

## (b) 单层重计算(不嵌套)下的最优分块 — 实测

**硬件**:RTX PRO 6000 96GB。**配置**:xl, batch=4, seq=2048, forward+backward, 3 steps。
把 32 个 block 按 `group_size` 分组,每组一次 `torch.utils.checkpoint`(单层,不嵌套)。

| group_size | 峰值显存 (MiB) | 说明 |
|---|---|---|
| 0(无 checkpoint) | **OOM**(>95 GB) | 32 层激活全留,放不下 |
| **1**(每块 checkpoint) | **39324** | ✅ 最低 |
| 2 | 45421 | |
| 4 | 57617 | |
| 8 | 82008 | |
| 16 | OOM(>95 GB) | |

**结论:`group_size = 1`(把每一个 TransformerBlock 单独 checkpoint)峰值最低,且峰值随分块增大单调上升。**

原因:本模型单个 block 的 residual **M**(主要是注意力 seq×seq 分数矩阵,见 2.1.6(e)(f))
远大于 checkpoint 在分界保存的输入 **a**(≈ batch×seq×d_model)。单层方案峰值 ≈ (N/s)·a + s·M,
因 a ≪ M,该式在 s=1 处取最小 —— 即**分块越细越省**,直到每块各自一个 checkpoint。
所以与"取 √N 分组"的通用结论不同:当 residual 由注意力主导时,**最优单层策略就是逐块 checkpoint**
(也正是工程实践中 transformer 的标准做法)。

**与相邻分块对比验证**:g=1(39.3 GiB)< g=2(44.4)< g=4(56.3)< g=8(80.1)< g=16/none(OOM),
单调关系成立;g=1 已是最细粒度(再细需嵌套,属 (a) 的范畴),故为单层最优。
相比无 checkpoint(OOM,理论 ~114 GiB 激活),逐块 checkpoint 把峰值压到 ~38 GiB,大幅腾出显存。

## 复现

```bash
cd assignment2-systems
for g in 0 1 2 4 8 16; do
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
    cs336_systems/benchmark/checkpoint_experiment.py --group_size $g --batch 4 --seq 2048 --steps 3
done
```
