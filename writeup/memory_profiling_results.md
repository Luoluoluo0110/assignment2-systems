# 2.1.6 Memory Profiling — xl 模型显存分析

**硬件**: NVIDIA RTX PRO 6000 Blackwell, **96 GB**, driver 595.71.05 / CUDA 13.2
**环境**: uv.lock 环境,**torch 2.11.0+cu130**, Python 3.12
**模型**: xl (d_model=2560, d_ff=10240, num_layers=32, num_heads=32, vocab=32000, **3.52B params**)
**设置**: batch=1, profile_step=3, warmup=5
**精度**: fp32 = `nullcontext()`；bf16 = `torch.autocast(bfloat16)`

---

## (b) 峰值显存表 (MiB, batch=1)

| context length | forward | full training step |
|---|---|---|
| 128  | fp32 **13600** / bf16 **20137** | fp32 **55428** / bf16 **55418** |
| 2048 | fp32 **15984** / bf16 **21576** | fp32 **94847*** / bf16 **86045** |

\* fp32 ctx2048 full_train 默认会因显存碎片 OOM(略超 96GB),加
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 后实测 94847 MiB(92.6 GiB)。

参考(40GB 卡上 forward_backward,不含 optimizer):ctx128 ≈ 27326 MiB,ctx2048 OOM。

## (a) 时间线(memory_viz)

snapshot pickle 全部生成在 `snapshots/`(8 个:forward/full_train × ctx{128,2048} × {fp32,bf16}),
拖进 https://pytorch.org/memory_viz 看 Active Memory Timeline。**能从峰值形状认出阶段**:

- **forward(纯推理)**:显存先随权重加载到 ~13 GiB 平台,之后每步只有少量瞬时激活的小起伏,整体平稳。
- **full training step**:呈现明显三段 —— ① forward 激活单调爬升;② backward 阶段激活逐步释放、梯度同时产生(锯齿下降);③ `optimizer.step()` 时一次性分配 Adam 的 m/v,出现一个**新的大台阶**(常驻 26 GiB),这是 full_train 与 forward 峰值差异的主因。

## (c) 混合精度对显存的影响(关键发现:取决于是否激活主导)

| 场景 | fp32 | bf16 | bf16 效果 |
|---|---|---|---|
| forward ctx128 | 13600 | 20137 | **+48%(更高)** |
| forward ctx2048 | 15984 | 21576 | **+35%(更高)** |
| full_train ctx128 | 55428 | 55418 | ≈ 持平 |
| full_train ctx2048 | 94847 | 86045 | **−9%(更低,省 ~8.6 GiB)** |

**结论:混合精度对显存的影响不是单调的,取决于激活在总显存中的占比。**
1. **forward / 短 ctx**:autocast 不改权重 dtype,fp32 主副本(13 GiB)常驻,还**额外缓存权重的 bf16 副本** → bf16 反而更高。
2. **full_train ctx128**:大头是 fp32 权重+梯度+Adam(52 GiB,与精度无关),激活极小 → fp32≈bf16。
3. **full_train ctx2048**:激活变大,bf16 把激活减半省下 ~8.6 GiB → 这里 bf16 才真正有用(默认配置下甚至是 fp32 OOM、bf16 能跑的分水岭)。
> 所以「混合精度省显存」只在**激活主导**时成立;权重/优化器状态占主导时基本没用。

## (d) 残差流单个激活张量大小 (fp32)

残差流张量形状 = `(batch, seq_len, d_model)`,fp32 每元素 4 字节:

```
bytes = batch × seq_len × d_model × 4，再 / 1024² 得 MiB
xl, batch=1, d_model=2560:
  ctx 128 : 1×128×2560×4  = 1,310,720 B  = 1.25 MiB
  ctx 2048: 1×2048×2560×4 = 20,971,520 B = 20.0 MiB
```

## 训练态显存拆解(解释各峰值)

```
xl = 3.52B params
权重 fp32 13.1 GiB + 梯度 13.1 + Adam m,v 26.2 = 52.4 GiB(常驻,与 ctx 无关)
full_train ctx128  ≈ 52.4 + 激活~1.8 + 临时 ≈ 55.4 GiB  ✓
full_train ctx2048 ≈ 52.4 + 激活~38(fp32) + logits/临时 ≈ 94.8 GiB  ✓
```

## (e) 最大分配来源

把 Detail 降到 ~10% 后,最大的分配是 **511.5 MiB**(536,346,752 字节)。它是注意力分数矩阵
`[batch, heads, seq, seq] = [1, 32, 2048, 2048]`(1×32×2048×2048×4 B = 512 MiB)。
stack trace 指向 `cs336_basics/model.py:431` 的 `scaled_dot_product_attention`,
具体由 `torch.where`(施加因果掩码,softmax 之前)分配:

```
at::native::where  →  model.py:431  scaled_dot_product_attention
                  →  model.py:524  forward (attention)
                  →  model.py:382  forward (TransformerBlock)
```

即朴素注意力为每个 head 显式 materialize 完整的 seq×seq 掩码后分数矩阵,随序列长度**平方增长** ——
这正是 §4 FlashAttention 要消除的开销。

## (f) 单个 TransformerBlock 的激活与梯度

用 `torch.autograd.graph.saved_tensors_hooks` 测得单个 xl TransformerBlock(batch=1, seq=2048)
**为反向保存的激活 = 2723 MiB**,按产生它的 op 归类,5 大贡献:

| 排名 | op (grad_fn) | 大小 | 占比 | 来源 |
|---|---|---|---|---|
| 1 | ExpBackward0 | 1024 MiB | 37.6% | softmax 的 exp(注意力概率 seq×seq×heads,2 份) |
| 2 | ViewBackward0 | 892 MiB | 32.8% | 注意力分数/QKV 的 view |
| 3 | ReshapeAliasBackward0 | 440 MiB | 16.2% | reshape |
| 4 | SigmoidBackward0 | 160 MiB | 5.9% | SwiGLU 的 SiLU(x·sigmoid) |
| 5 | MulBackward0 | 120 MiB | 4.4% | 逐元素乘 |

前三项(Exp+View+Reshape,占 ~86%)全是注意力相关 —— 与 (e) 一致,注意力是激活显存大头。
（注:View/Reshape 是别名视图,可能与底层存储共享,求和略有重复计;但作为"贡献 op"如实列出。）

**梯度张量大小**:单个 block 参数 = 104.86M → 梯度 = 104.86M × 4B = **400 MiB**(每参数一个同形状梯度)。

**反向时每个 block 的显存变化**:反向到某个 block 时,该 block 保存的 ~2723 MiB 激活被释放,
同时产生 ~400 MiB 梯度 → 净显存**下降** ~2300 MiB/block(锯齿下降)。由此反推梯度 ≈ 释放的激活 − 净下降,
得 ~400 MiB,**与参数量×4B 完全吻合,符合预期**(autograd 对每个参数产生一个同形状梯度)。

**Nsight Systems 截图**:`block_mem.nsys-rep`(`nsys profile --trace=cuda,nvtx --cuda-memory-usage=true`
+ 每个 TransformerBlock 的 NVTX 标签)。GUI 时间线可见周期性的 `forward → backward → optimizer`
及 forward 内的 `block0..block31` 子范围。

> 注:本机 nsys 为 2024.1.1(nsight-compute 自带),其 CUPTI 在 Blackwell/CUDA 13.2 上初始化失败
> (`CUPTI_ERROR_NOT_INITIALIZED`),故 **GPU 显存轨道未采集**,只有 NVTX(CPU 侧)有效。
> 因此「逐 block 显存变化」改由 **PyTorch memory_viz 的 full_train 快照**佐证:放大 backward 段可见
> 峰顶往下 32 个小台阶的锯齿,每个台阶对应一个 TransformerBlock 反向时释放其 ~2723 MiB(ctx2048)激活、
> 同时产生 ~400 MiB 梯度,净显存逐 block 下降。定量数字以上面 `saved_tensors_hooks` 测量为准。

---

## 复现(uv 环境,Blackwell 服务器)

```bash
cd assignment2-systems
.venv/bin/python cs336_systems/benchmark/mem_sweep.py
# fp32 ctx2048 full_train 需防碎片：
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True .venv/bin/python \
  cs336_systems/benchmark/benchmark_script.py --model_name xl --context_length 2048 \
  --mode full_train --memory_profile --profile_step 3 --snapshot_dir snapshots
```
