# Nsight Systems Profiling Report (nsys_profile)

## Setup

| Item | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4090 (24 GB) |
| Nsight Systems | 2024.6.2 |
| Precision / batch | FP32, batch size = 1 |
| Steps per profile | 2 warm-up + 3 timed |
| nsys command | `nsys profile --trace=cuda,nvtx ... uv run python ... benchmark_script.py` |

**Two model sizes** (from Table 1) and **three power-of-two context lengths > 128**:

| Model | d_model | d_ff | layers | heads | d_head |
|---|---|---|---|---|---|
| `small` | 768 | 3072 | 12 | 12 | 64 |
| `large` | 1280 | 5120 | 36 | 20 | 64 |

Context lengths: **256, 512, 1024**. 1024 is the largest power-of-two that fits in 24 GB
(see the memory note under (d): `large` full-training at 1024 OOMs).

**Methodology notes**
- A `torch.cuda.synchronize()` is placed *inside* each NVTX range (`forward`, `loss`,
  `backward`, `optimizer.step`) so the GPU kernels align with the CPU-side range window
  and per-pass attribution is correct for every timed step.
- Per-pass *forward* numbers come from the `:forward` NVTX range (exactly the 3 timed
  steps). Whole-pass fractions for forward / forward+backward / full-train come from the
  **CUDA GPU Kernel Summary** of each run (warm-up does identical work, so the fraction is
  representative). Matmul = `*sgemm*` kernels.
- Self-attention is sub-annotated with `attn_scores_QK`, `attn_softmax`, `attn_AV` ranges
  for question (e).

---

## (a) Total time of the forward pass — does it match the Python stdlib timing?

| Model | Context | nsys forward (ms) | Python stdlib (ms) | ratio |
|---|---|---|---|---|
| small | 256  | 18.92  | 19.00  | 1.00 |
| small | 512  | 19.17  | 19.20  | 1.00 |
| small | 1024 | 21.77  | 21.90  | 0.99 |
| large | 256  | 56.09  | 56.20  | 1.00 |
| large | 512  | 57.07  | 57.10  | 1.00 |
| large | 1024 | 111.78 | 112.30 | 1.00 |

**Answer.** The forward pass ranges from ~19 ms (`small`) to ~112 ms (`large`, ctx 1024),
and the nsys `forward` NVTX-range time matches the Python-stdlib measurement to within **1 %
in every configuration** — expected, since both bracket exactly the same GPU work and we
synchronize at the end of the range.

---

## (b) Which CUDA kernel takes the most GPU time, and how many times is it called?

| Model | Context | Top kernel in **forward** | calls / forward | Top kernel in **forward+backward** | calls / step |
|---|---|---|---|---|---|
| small | 256  | `ampere_sgemm_128x64_tn` (5.4 ms)  | 37  | *elementwise (mask/scale/copy)* (11.4 ms) | 157 |
| small | 512  | `ampere_sgemm_64x64_tn` (8.4 ms)   | 60  | `ampere_sgemm_64x64_nn` (18.9 ms)         | 72  |
| small | 1024 | `ampere_sgemm_128x64_tn` (20.4 ms) | 84  | `ampere_sgemm_128x64_tn` (34.1 ms)        | 84  |
| large | 256  | `ampere_sgemm_128x64_tn` (44.5 ms) | 253 | *elementwise (mask/scale/copy)* (76.9 ms) | 469 |
| large | 512  | `ampere_sgemm_128x64_tn` (88.5 ms) | 253 | `ampere_sgemm_128x64_tn` (149.0 ms)       | 253 |
| large | 1024 | `ampere_sgemm_128x64_tn` (166.7 ms)| 252 | `ampere_sgemm_128x64_tn` (279.4 ms)       | 252 |

**Answer.** In the forward pass the most cumulative GPU time is always an FP32 matrix-multiply
kernel — `ampere_sgemm_128x64_tn` — invoked e.g. ~253× per forward for `large` (the GEMM used
by the Linear/FFN projections). For forward **and** backward it stays the top kernel at ctx ≥ 512
(now ~2× the time, since backward does a second matmul), **except at the shortest context (256)**,
where backward launches so many tiny memory-bound elementwise kernels that a single
masking/scaling elementwise kernel overtakes the GEMM.

---

## (c) What non-matmul kernels take non-trivial runtime in the forward pass?

Share of total forward GPU time (representative ctx = 512):

| Kernel category | small ctx512 | large ctx512 |
|---|---|---|
| **matmul (sgemm)** | 77.5 % | 81.4 % |
| elementwise (mask / scale-by-1/√d / residual add / RoPE apply) | 15.1 % | 12.6 % |
| softmax (max + exp + sum reductions) | 2.4 % | 2.2 % |
| RMSNorm (mean reduction + rsqrt + pow) | 2.0 % | 1.3 % |
| RoPE concat (`CatArrayBatchedCopy`) | 1.3 % | 0.9 % |
| other reductions | 0.9 % | 0.8 % |
| SiLU activation (`sigmoid`) | 0.7 % | 0.7 % |

**Answer.** Aside from the GEMMs, the non-trivial kernels are all memory-bound pointwise/
reduction ops: generic **elementwise** kernels (masking, the 1/√d scaling, residual adds and
RoPE) dominate at **~13–15 %**, followed by **softmax** reductions (~2–2.5 %), **RMSNorm**
reductions (~1.5–2 %), **RoPE** concat (~1 %) and the **SiLU** activation (~0.7 %).

---

## (d) How does the matmul fraction change from inference to a full training step?

Share of GPU time spent in matmul kernels:

| Model | Context | forward only | forward + backward | full training step (incl. AdamW) |
|---|---|---|---|---|
| small | 256  | 73.4 % | 58.7 % | 29.6 % |
| small | 512  | 77.5 % | 62.8 % | 40.3 % |
| small | 1024 | 60.0 % | 47.9 % | 38.3 % |
| large | 256  | 77.0 % | 62.7 % | 28.1 % |
| large | 512  | 81.4 % | 66.7 % | 38.8 % |
| large | 1024 | 61.3 % | 51.0 % | **OOM** (training does not fit in 24 GB) |

> Memory note: `large` @ ctx 1024 runs **forward** (and even forward+backward) fine, but a full
> training step OOMs during backward because AdamW's optimizer state adds 2× the parameter
> memory. So 1024 is the largest training context for `small`, while for `large` it is 512.

**Answer.** Going from inference to a full training step the matmul share roughly **halves**
(from ~73–81 % down to ~28–44 %): the backward pass adds many elementwise gradient kernels
(matmul drops to ~50–67 %) and AdamW's `optimizer.step` is **entirely** elementwise, so the
memory-bound (non-matmul) kernels' share correspondingly roughly **doubles** and dominates the
training step.

---

## (e) Softmax vs. matmul inside self-attention — runtime vs. FLOPs

Per-forward GPU time of the softmax vs. the two attention matmuls (QK·Kᵀ and attn·V):

| Model | Context | softmax (ms) | attn matmul QK+AV (ms) | runtime ratio matmul/softmax | FLOP ratio matmul/softmax |
|---|---|---|---|---|---|
| small | 256  | 0.21  | 0.40 | 1.89× | ~51× |
| small | 512  | 0.47  | 0.76 | 1.64× | ~51× |
| small | 1024 | 2.87  | 1.89 | **0.66×** | ~51× |
| large | 256  | 0.83  | 1.27 | 1.54× | ~51× |
| large | 512  | 2.17  | 2.81 | 1.29× | ~51× |
| large | 1024 | 23.77 | 9.43 | **0.40×** | ~51× |

FLOP estimate per layer (batch 1): attention matmuls ≈ `4·seq²·d_model`, softmax ≈ `5·n_heads·seq²`,
so the FLOP ratio ≈ `(4/5)·d_head = 0.8·64 ≈ 51×` (same for both models since d_head = 64).

**Answer.** The two attention matmuls carry **~50× more FLOPs** than the softmax, yet they take
**at most ~2×** the runtime — and at ctx = 1024 the softmax actually runs **1.5–2.5× longer**
than the matmuls. Softmax is therefore heavily **memory-bandwidth bound** (it streams the whole
seq×seq score matrix several times for the max/exp/sum/div), whereas the GEMMs are compute-
efficient; the mismatch between runtime and FLOPs grows with context length because the softmax's
O(seq²) data movement scales faster than the hardware can hide it.

---

## Key takeaways

1. nsys NVTX-range timing reproduces the Python wall-clock forward time to within 1 %.
2. The forward pass is GEMM-dominated (~75–80 % of GPU time); a single `ampere_sgemm` kernel
   is the top consumer.
3. Memory-bound pointwise/reduction kernels (elementwise, softmax, RMSNorm, RoPE, SiLU) are
   individually small but together ~20 % of forward, and they **dominate** a full training step
   once backward + AdamW are included (matmul share halves to ~30–40 %).
4. Softmax's runtime is wildly disproportionate to its FLOPs and overtakes the attention
   matmuls at long context — the classic motivation for a fused, IO-aware attention (FlashAttention).
5. 24 GB limits full training of `large` to ctx ≤ 512 (forward alone reaches 1024).
