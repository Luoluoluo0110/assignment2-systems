"""跑全套混合精度对比实验：5 个模型尺寸 × {fp32, bf16}，分别测 forward / backward。

用法（在仓库根目录）:
    PYTHONPATH=cs336-basics:. python cs336_systems/benchmark/run_sweep.py

依次以子进程调用 benchmark_script.py（每个组合独立进程，避免显存累积），
解析其 RESULT 行，最后打印 Markdown 对比表（含 bf16 相对 fp32 的加速比）。
"""
import os
import re
import subprocess
import sys

MODELS = ["small", "medium", "large", "xl", "10B"]
MODE = "forward_backward"
WARMUP = 5
STEPS = 10

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "benchmark_script.py")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))

RESULT_RE = re.compile(r"RESULT\s+(.*)")


def parse_result(line):
    fields = {}
    for kv in line.split():
        if "=" in kv:
            k, v = kv.split("=", 1)
            fields[k] = v
    return fields


def run_one(model, bf16):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(REPO_ROOT, "cs336-basics"), REPO_ROOT])
    cmd = [sys.executable, SCRIPT, "--model_name", model, "--mode", MODE,
           "--warmup_steps", str(WARMUP), "--test_steps", str(STEPS)]
    if bf16:
        cmd.append("--bf16")
    label = f"{model:>6} | {'bf16' if bf16 else 'fp32'}"
    print(f">>> running {label} ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if proc.returncode != 0:
        if "out of memory" in out.lower() or "CUDA out of memory" in out:
            print(f"    {label}: OOM")
            return {"status": "OOM"}
        print(f"    {label}: FAILED (exit {proc.returncode})")
        print("    " + out.strip().splitlines()[-1] if out.strip() else "")
        return {"status": "FAIL"}
    m = None
    for ln in out.splitlines():
        mm = RESULT_RE.search(ln)
        if mm:
            m = mm
    if not m:
        print(f"    {label}: no RESULT line")
        return {"status": "NORESULT"}
    fields = parse_result(m.group(1))
    fields["status"] = "ok"
    print(f"    {label}: fwd={float(fields['fwd_mean'])*1000:.1f}ms "
          f"bwd={float(fields['bwd_mean'])*1000:.1f}ms "
          f"tot={float(fields['tot_mean'])*1000:.1f}ms")
    return fields


def fmt_ms(fields, key):
    if fields.get("status") != "ok":
        return fields.get("status", "-")
    return f"{float(fields[key + '_mean'])*1000:.1f}±{float(fields[key + '_std'])*1000:.1f}"


def speedup(fp32, bf16, key):
    if fp32.get("status") != "ok" or bf16.get("status") != "ok":
        return "-"
    a = float(fp32[key + "_mean"])
    b = float(bf16[key + "_mean"])
    if b == 0:
        return "-"
    return f"{a / b:.2f}x"


def main():
    results = {}  # (model, prec) -> fields
    for model in MODELS:
        for bf16 in (False, True):
            results[(model, "bf16" if bf16 else "fp32")] = run_one(model, bf16)

    print("\n\n## 结果汇总 (forward_backward, batch=1, seq=512, warmup=%d, steps=%d)\n" % (WARMUP, STEPS))
    print("时间单位 ms (mean±std)。speedup = fp32 / bf16。\n")
    header = ("| size | fwd fp32 | fwd bf16 | fwd↑ | bwd fp32 | bwd bf16 | bwd↑ "
              "| total fp32 | total bf16 | total↑ |")
    sep = "|" + "|".join(["---"] * 10) + "|"
    print(header)
    print(sep)
    for model in MODELS:
        fp = results[(model, "fp32")]
        bf = results[(model, "bf16")]
        row = [
            model,
            fmt_ms(fp, "fwd"), fmt_ms(bf, "fwd"), speedup(fp, bf, "fwd"),
            fmt_ms(fp, "bwd"), fmt_ms(bf, "bwd"), speedup(fp, bf, "bwd"),
            fmt_ms(fp, "tot"), fmt_ms(bf, "tot"), speedup(fp, bf, "tot"),
        ]
        print("| " + " | ".join(row) + " |")


if __name__ == "__main__":
    main()
