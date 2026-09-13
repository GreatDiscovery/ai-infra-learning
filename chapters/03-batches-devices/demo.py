"""对同一批样本、同一模型，改变推理 batch 与输入搬运方式。"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _support import build_model, choose_device, measure_variants, output_dir, save


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--samples", type=int, default=512)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16, 64, 256])
    p.add_argument("--repeats", type=int, default=5)
    a = p.parse_args()
    if not (64 <= a.samples <= 4096 and 3 <= a.repeats <= 30):
        p.error("samples 需为 64–4096；repeats 需为 3–30")
    if not 1 <= len(a.batches) <= 8 or any(not 1 <= b <= a.samples for b in a.batches):
        p.error("指定 1–8 个 batch size，每个需在 1 到 samples 之间")
    try:
        chosen = choose_device(a.device)
    except ValueError as e:
        p.error(str(e))
    devices = ["cpu", chosen] if a.device == "auto" and chosen != "cpu" else [chosen]
    torch.manual_seed(42)
    torch.set_num_threads(4)
    images = torch.rand(a.samples, 1, 28, 28)
    reference_model = build_model().eval()
    reference = reference_model(images)
    results = []
    print("使用随机图片和未训练模型，只研究系统行为，不评价分类准确率。")
    for device in devices:
        model = build_model().to(device).eval()
        model.load_state_dict(reference_model.state_dict())
        resident = images.to(device)
        for batch in a.batches:
            cpu_parts, device_parts = images.split(batch), resident.split(batch)

            def per_batch():
                return torch.cat([model(part.to(device)) for part in cpu_parts])

            def already_resident():
                return torch.cat([model(part) for part in device_parts])

            variants = {"per_batch": per_batch, "resident": already_resident}
            for fn in variants.values():
                torch.testing.assert_close(fn().cpu(), reference, rtol=1e-3, atol=1e-4)
            measured = measure_variants(variants, device, a.repeats)
            for mode, timing in measured.items():
                ms = timing["median_ms"]
                row = {"device": device, "batch": batch, "mode": mode,
                       "calls": len(cpu_parts), "samples_per_second": a.samples * 1000 / ms, **timing}
                results.append(row)
                print(f"{device:4s} batch={batch:4d} {mode:9s} | "
                      f"{ms:8.3f} ms / {a.samples} 张 | {row['samples_per_second']:9.1f} 张/s", flush=True)
    lines = ["# 03 · batch 与设备", "", f"参数：`{vars(a)}`。FP32，CPU 线程 4。", "",
             "| 设备 | batch | 模式 | 前向调用数 | 完成整组 ms | 张/s |",
             "|---|---:|---|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['device']} | {r['batch']} | {r['mode']} | {r['calls']} | "
                     f"{r['median_ms']:.3f} | {r['samples_per_second']:.1f} |")
    lines += ["", "每种配置预热 3 次，重复测量取中位数；两种输入模式交替先后顺序。所有结果与同一 CPU 参考输出比较通过。",
              "", "`per_batch` 将每一批输入在计时内送到目标设备；`resident` 在计时前已把全部输入放到设备。"
              "两者均计入前向、结果拼接及最终同步；不计入数据生成、模型加载和结果拷回 CPU。",
              "", "CPU 上 `.to('cpu')` 对同类型 CPU 张量通常无需复制，不能把两种模式的差值称为 CPU 搬运耗时。"
              "Apple 使用统一内存，但后端仍可能存在分配、复制和同步开销。",
              "", "完成整组时间除以样本数只是平均摊销成本，不是线上单请求延迟。resident 的输入内存占用随样本数增长；"
              "真实服务的凑批等待、排队和输出回传均未模拟。",
              "", "练习：固定设备，比较 batch=1 与 batch=64 的调用数和吞吐；再固定 batch 比较设备与输入模式。"
              "记录观察，避免把噪声或多个同时改变的因素当成单一原因。"]
    save(output_dir("03-batches-devices"), {"config": vars(a), "results": results}, lines)


if __name__ == "__main__":
    main()
