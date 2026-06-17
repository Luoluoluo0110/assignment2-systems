"""xl 模型显存分析 sweep：ctx{128,2048} × mode{forward,full_train} × {fp32,bf16}。

每个组合独立子进程调用 benchmark_script.py --memory_profile，
解析 [MEM ...] peak= 行；OOM 则标记 OOM。最后打印 Markdown 峰值表。
快照 pickle 统一落在 snapshots/ 目录。
"""
import os
import re
import subprocess
import sys

CTXS = [128, 2048]
MODES = ["forward", "full_train"]
PRECS = [False, True]   # False=fp32, True=bf16

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "benchmark_script.py")
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SNAP_DIR = os.path.join(REPO_ROOT, "snapshots")

PEAK_RE = re.compile(r"peak=([0-9.]+)\s*MiB")


def run_one(cl, mode, bf16):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(REPO_ROOT, "cs336-basics"), REPO_ROOT])
    cmd = [sys.executable, SCRIPT, "--model_name", "xl", "--context_length", str(cl),
           "--mode", mode, "--memory_profile", "--profile_step", "3",
           "--snapshot_dir", SNAP_DIR]
    if bf16:
        cmd.append("--bf16")
    label = f"xl ctx{cl} {mode} {'bf16' if bf16 else 'fp32'}"
    print(f">>> {label} ...", flush=True)
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    out = proc.stdout + proc.stderr
    if "out of memory" in out.lower():
        print(f"    {label}: OOM")
        return "OOM"
    m = PEAK_RE.search(out)
    if not m:
        print(f"    {label}: FAIL ({proc.returncode})")
        tail = out.strip().splitlines()[-1] if out.strip() else ""
        print("    " + tail)
        return "FAIL"
    peak = float(m.group(1))
    print(f"    {label}: peak={peak:.1f} MiB")
    return peak


def cell(v):
    if isinstance(v, float):
        return f"{v:.1f}"
    return v


def main():
    res = {}
    for cl in CTXS:
        for mode in MODES:
            for bf16 in PRECS:
                res[(cl, mode, bf16)] = run_one(cl, mode, bf16)

    print("\n\n## xl 峰值显存 (MiB, batch=1, profile_step=3)\n")
    print("| context length | forward fp32 | forward bf16 | full_train fp32 | full_train bf16 |")
    print("|---|---|---|---|---|")
    for cl in CTXS:
        row = [str(cl),
               cell(res[(cl, "forward", False)]), cell(res[(cl, "forward", True)]),
               cell(res[(cl, "full_train", False)]), cell(res[(cl, "full_train", True)])]
        print("| " + " | ".join(row) + " |")
    print(f"\n快照目录: {SNAP_DIR}")


if __name__ == "__main__":
    main()
