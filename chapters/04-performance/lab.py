"""Mac AI Infra 入门实验：预测 → 测量 → 解释。无需下载模型。"""

import argparse
import csv
import json
import platform
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

import torch


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def measure(fn, device, repeats):
    """每次操作完成后计时：包含 Python、调度和同步成本，非纯 kernel 时间。"""
    for _ in range(5):
        fn()
    sync(device)
    samples = []
    for _ in range(repeats):
        sync(device)
        start = time.perf_counter_ns()
        fn()
        sync(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "samples_ms": samples,
    }


def verify(actual, expected, label):
    # 不同设备/矩阵内核的浮点累加顺序可能不同；这里只做结果一致性检查。
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=1e-3, atol=1e-3)
    print(f"  正确性检查通过：{label}", flush=True)


def shape_experiment(devices, args):
    print("\n实验 1：增大 M，会让耗时也按比例增长吗？", flush=True)
    print("  A[M,K] @ B[K,N]，固定 K=N；张量已在目标设备上。")
    print(f"  {'设备':<7} {'M':>5} {'中位数 ms':>12} {'GFLOP/s':>12} {'估算 FLOP/B':>13}")
    rows = []
    sizes = [1, 8, 32, 128, 512]
    # 固定随机顺序，减轻总是由小到大测量带来的温度/频率顺序偏差。
    random.Random(42).shuffle(sizes)
    for m in sizes:
        a = torch.randn(m, args.width)
        b = torch.randn(args.width, args.width)
        reference = a @ b
        for device in devices:
            x, w = a.to(device), b.to(device)
            out = torch.empty((m, args.width), device=device)
            fn = lambda: torch.mm(x, w, out=out)
            fn()
            verify(out, reference, f"{device}, M={m}")
            timing = measure(fn, device, args.repeats)
            flops = 2 * m * args.width * args.width
            # 每个输入读一次、输出写一次的理想字节量，不是实测 DRAM 流量。
            ideal_bytes = 4 * (m * args.width + args.width**2 + m * args.width)
            row = {
                "experiment": "shape", "device": device, "m": m,
                "k": args.width, "n": args.width,
                "flops": flops, "ideal_bytes": ideal_bytes,
                "arithmetic_intensity_estimate": flops / ideal_bytes,
                "gflops": flops / (timing["median_ms"] * 1e6), **timing,
            }
            rows.append(row)
            print(f"  {device:<7} {m:>5} {row['median_ms']:>12.4f} "
                  f"{row['gflops']:>12.2f} {flops / ideal_bytes:>13.2f}", flush=True)
    return rows


def batch_experiment(devices, args):
    print("\n实验 2：同样处理 64 行，逐行调用与一次批量调用有什么区别？", flush=True)
    rows = []
    a = torch.randn(64, args.width)
    b = torch.randn(args.width, args.width)
    reference = a @ b
    for device in devices:
        x, w = a.to(device), b.to(device)
        loop_out = torch.empty_like(x)
        batch_out = torch.empty_like(x)
        # 在计时外准备切片，避免把重复创建 view 当作主要差异。
        slices = [(x[i:i+1], loop_out[i:i+1]) for i in range(64)]

        def loop():
            for src, dst in slices:
                torch.mm(src, w, out=dst)

        def batch():
            torch.mm(x, w, out=batch_out)

        loop()
        batch()
        verify(loop_out, reference, f"{device}, 逐行计算")
        verify(batch_out, reference, f"{device}, 批量计算")
        for name, fn in [("row_loop", loop), ("batch", batch)]:
            timing = measure(fn, device, args.repeats)
            rows.append({"experiment": "batch", "device": device,
                         "mode": name, "rows": 64, **timing})
            print(f"  {device:>3} {name:>8}: {timing['median_ms']:.4f} ms / 64 行", flush=True)
    return rows


def async_experiment(args, device):
    print("\n实验 3：GPU 调用返回时，计算真的完成了吗？", flush=True)
    x = torch.randn(args.width, args.width, device=device)
    w = torch.randn_like(x)
    out = torch.empty_like(x)
    for _ in range(5):
        torch.mm(x, w, out=out)
    sync(device)
    samples = []
    for _ in range(args.repeats):
        sync(device)
        start = time.perf_counter_ns()
        for _ in range(10):
            torch.mm(x, w, out=out)
        submitted = time.perf_counter_ns()
        sync(device)
        finished = time.perf_counter_ns()
        samples.append({
            "submit_ms": (submitted - start) / 1e6,
            "total_ms": (finished - start) / 1e6,
            "wait_ms": (finished - submitted) / 1e6,
        })
    row = {"experiment": "async", "device": device, "operations": 10,
           "submit_ms": statistics.median(s["submit_ms"] for s in samples),
           "total_ms": statistics.median(s["total_ms"] for s in samples),
           "wait_ms": statistics.median(s["wait_ms"] for s in samples),
           "samples": samples}
    print(f"  10 次 matmul：仅量调用返回 {row['submit_ms']:.3f} ms；"
          f"包含完成等待 {row['total_ms']:.3f} ms。", flush=True)
    print("  提交期间 CPU 与 GPU 可能重叠工作；差值不能当作纯 GPU 执行时间。")
    return [row]


def save_report(rows, metadata, output):
    output.mkdir(parents=True, exist_ok=False)
    (output / "results.json").write_text(
        json.dumps({"metadata": metadata, "results": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    flattened = [{k: v for k, v in row.items() if not isinstance(v, list)} for row in rows]
    fields = list(dict.fromkeys(k for row in flattened for k in row))
    with (output / "results.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)

    lines = ["# 本机实验结果", "", f"时间：{metadata['time']}  ",
             f"系统：{metadata['platform']}  ",
             f"PyTorch：{metadata['torch']}；设备：{', '.join(metadata['devices'])}；"
             f"CPU intra-op 线程：{metadata['threads']}；FP32；K=N={metadata['width']}。", "",
             "每组预热 5 次，报告重复测量的中位数；原始样本见 results.json。",
             "耗时包含 Python 调用、设备调度及完成同步；不包含输入生成、设备搬运及正确性检查。", "",
             "## 1. 矩阵形状", "",
             "| 设备 | M | 中位数 ms | 最小–最大 ms | GFLOP/s | 理想 FLOP/B |",
             "|---|---:|---:|---:|---:|---:|"]
    shapes = [r for r in rows if r["experiment"] == "shape"]
    for r in sorted(shapes, key=lambda r: (r["m"], r["device"])):
        lines.append(f"| {r['device']} | {r['m']} | {r['median_ms']:.4f} | "
                     f"{r['min_ms']:.4f}–{r['max_ms']:.4f} | {r['gflops']:.2f} | "
                     f"{r['arithmetic_intensity_estimate']:.2f} |")
    lines += ["", "先观察：M 从 1 到 512，计算量增加 512 倍，耗时增加了多少倍？", ""]
    for device in metadata["devices"]:
        small = next(r for r in shapes if r["device"] == device and r["m"] == 1)
        large = next(r for r in shapes if r["device"] == device and r["m"] == 512)
        lines.append(f"- {device}：本次耗时增加 **{large['median_ms']/small['median_ms']:.2f} 倍**，"
                     f"有效计算吞吐变化为 **{large['gflops']/small['gflops']:.2f} 倍**。")
    lines += ["", "解释线索：增大 M 可以提高权重复用并增加并行工作量；小任务也可能主要受调度延迟限制。",
              "FLOP/B 按输入各读一次、输出写一次估算。缓存、内核分块和重复访存会改变实际流量；"
              "不能仅凭此表断言某一项一定 memory-bound，也不能把估算字节量/耗时当成实测显存带宽。", "",
              "## 2. 逐行与批量", "", "| 设备 | 逐行 ms | 批量 ms | 逐行耗时 / 批量耗时 |",
              "|---|---:|---:|---:|"]
    batches = [r for r in rows if r["experiment"] == "batch"]
    for device in metadata["devices"]:
        loop = next(r for r in batches if r["device"] == device and r["mode"] == "row_loop")
        batch = next(r for r in batches if r["device"] == device and r["mode"] == "batch")
        lines.append(f"| {device} | {loop['median_ms']:.4f} | {batch['median_ms']:.4f} | "
                     f"{loop['median_ms']/batch['median_ms']:.2f}× |")
    lines += ["", "两种方式完成相同的 64 行线性计算，并已与 CPU 参考结果对比。比值大于 1 表示批量更快。",
              "这里同时改变了调用次数、矩阵形状和数据复用，不能把全部收益归因于 Python 循环。"
              "真实服务还需要考虑凑批等待、排队与尾延迟；本实验没有模拟这些成本。", "",
              "## 3. 异步计时", ""]
    async_rows = [r for r in rows if r["experiment"] == "async"]
    if async_rows:
        r = async_rows[0]
        lines += [f"10 次 GPU 矩阵乘法：只计调用返回 **{r['submit_ms']:.3f} ms**；"
                  f"包含完成等待 **{r['total_ms']:.3f} ms**（分别取中位数）。",
                  "GPU 调用可能异步返回。同步后的墙钟时间才包含工作完成，"
                  "但它仍包含主机开销，不等于纯 kernel 时间。"]
    else:
        lines.append("本次仅运行 CPU，未执行 GPU 异步计时实验。")
    lines += ["", "## 下一轮练习", "",
              "1. 在相同电源、后台负载下重复运行，查看结论是否稳定。",
              "2. 将 --width 从 1024 改为 2048，先预测结果，再测量。",
              "3. CPU 使用 --threads 1 与 --threads 4 分别运行；解释更多线程是否总有益。",
              "4. 面试练习：用现象、假设、证据、下一步验证讲述一次结果，区分观测与推断。", "",
              "这是热缓存、预分配输出的 FP32 微基准，不能直接代表完整 LLM、CUDA 或生产服务性能。",
              "M=1 与较大 M 只帮助理解单行与多行线性层计算；并未实现 Attention、KV Cache 或完整 Prefill/Decode。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda", "both"], default="auto")
    parser.add_argument("--width", type=int, choices=[256, 512, 1024, 2048], default=1024)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if not 3 <= args.repeats <= 100 or not 1 <= args.threads <= 16:
        parser.error("repeats 需为 3–100，threads 需为 1–16。")
    mps_available = torch.backends.mps.is_available()
    cuda_available = torch.cuda.is_available()
    gpu = "cuda" if cuda_available else ("mps" if mps_available else None)
    if args.device == "mps" and not mps_available:
        parser.error("MPS 不可用，可在支持的本机环境运行或选择 --device cpu。")
    if args.device == "cuda" and not cuda_available:
        parser.error("CUDA 不可用，请检查构建/驱动或选择 --device cpu。")
    if args.device == "both" and gpu is None:
        parser.error("当前无可用 GPU，请使用 --device cpu。")
    devices = (["cpu", gpu] if gpu else ["cpu"]) if args.device in ("auto", "both") else [args.device]
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    print(f"PyTorch {torch.__version__} | {platform.machine()} | MPS={mps_available} | CUDA={cuda_available}")
    print(f"设备：{devices} | CPU 线程：{torch.get_num_threads()} | FP32 | width={args.width}")
    if gpu is None and args.device == "auto":
        print("当前进程未检测到可用 GPU，自动运行 CPU 实验。")
    print("先预测，再看结果：GPU 一定更快吗？计算量增加 512 倍，耗时也会吗？", flush=True)
    with torch.inference_mode():
        rows = shape_experiment(devices, args)
        rows += batch_experiment(devices, args)
        for device in devices:
            if device != "cpu":
                rows += async_experiment(args, device)
    now = datetime.now().astimezone()
    metadata = {"time": now.isoformat(), "platform": platform.platform(),
                "torch": torch.__version__, "devices": devices, "mps_built": torch.backends.mps.is_built(),
                "mps_available": mps_available, "cuda_available": cuda_available, "threads": torch.get_num_threads(),
                "width": args.width, "repeats": args.repeats, "warmups": 5, "dtype": "float32"}
    output = Path(__file__).resolve().parents[2] / "results" / "performance" / now.strftime("%Y%m%d-%H%M%S-%f")
    save_report(rows, metadata, output)
    print(f"\n完成。中文报告：{(output / 'report.md').relative_to(Path(__file__).resolve().parents[2])}", flush=True)
    print("原始样本 results.json；汇总表 results.csv。")


if __name__ == "__main__":
    main()
