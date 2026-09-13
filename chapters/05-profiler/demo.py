"""给流水线分段，用 Profiler 定位重复的小算子，再验证向量化收益。"""
import argparse
from contextlib import nullcontext
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _support import ROOT, build_model, choose_device, measure_variants, output_dir, save, sync


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--repeats", type=int, default=10)
    a = p.parse_args()
    if not (8 <= a.batch_size <= 512 and 3 <= a.repeats <= 50):
        p.error("batch-size 需为 8–512；repeats 需为 3–50")
    try:
        device = choose_device(a.device)
    except ValueError as e:
        p.error(str(e))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    raw = torch.randint(256, (a.batch_size, 1, 28, 28), dtype=torch.uint8)
    model = build_model().to(device).eval()

    def pipeline(mode, instrument=False):
        label = record_function if instrument else lambda _: nullcontext()
        with label("01_prepare_input"):
            if mode == "row_loop":
                images = torch.stack([image.float() / 255.0 for image in raw])
            else:
                images = raw.float() / 255.0
        with label("02_to_device"):
            images = images.to(device)
        with label("03_forward"):
            result = model(images)
        with label("04_wait_complete"):
            sync(device)
        return result

    torch.testing.assert_close(pipeline("row_loop"), pipeline("vectorized"), rtol=1e-5, atol=1e-6)
    variants = {"row_loop": lambda: pipeline("row_loop"), "vectorized": lambda: pipeline("vectorized")}
    # 计时与 Profiler 分开：避免将 Profiler 的额外成本当成正常执行时间。
    timings = measure_variants(variants, device, a.repeats)
    path = output_dir("05-profiler")
    profiles = {}
    for mode in variants:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True, with_stack=False) as prof:
            for _ in range(3):
                pipeline(mode, instrument=True)
        events = prof.key_averages()
        (path / f"{mode}-operators.txt").write_text(
            events.table(sort_by="self_cpu_time_total", row_limit=15), encoding="utf-8")
        prof.export_chrome_trace(str(path / f"{mode}-trace.json"))
        stages = [{"name": e.key, "calls": e.count, "cpu_total_us": e.cpu_time_total,
                   "self_cpu_us": e.self_cpu_time_total}
                  for e in events if e.key.startswith(("01_", "02_", "03_", "04_"))]
        profiles[mode] = stages
    ratio = timings["row_loop"]["median_ms"] / timings["vectorized"]["median_ms"]
    print(f"逐行处理：{timings['row_loop']['median_ms']:.3f} ms；"
          f"向量化：{timings['vectorized']['median_ms']:.3f} ms；耗时比={ratio:.2f}×")
    lines = ["# 05 · 用证据定位瓶颈", "", f"设备：{device}；参数：`{vars(a)}`。", "",
             "## 未开启 Profiler 的完成时间", "", "| 方式 | 中位数 ms |", "|---|---:|"]
    for mode, r in timings.items():
        lines.append(f"| {mode} | {r['median_ms']:.4f} |")
    lines += ["", f"逐行/向量化耗时比为 **{ratio:.2f}×**，大于 1 才表示向量化更快。数值一致性检查通过。",
              "", "## Profiler 主机侧分段（3 次调用的合计）", "",
              "| 方式 | 阶段 | 次数 | CPU total μs | Self CPU μs |", "|---|---|---:|---:|---:|"]
    for mode, stages in profiles.items():
        for s in stages:
            lines.append(f"| {mode} | {s['name']} | {s['calls']} | {s['cpu_total_us']:.1f} | {s['self_cpu_us']:.1f} |")
    lines += ["", "先查看 `*-operators.txt` 中 `aten::div` 等小算子的调用次数，再对照 prepare_input 阶段。"
              "逐张转换改成整个 batch 一次转换，减少调用，同时产生相同模型输入。",
              "", "CPU total 包括子操作，Self CPU 排除子操作；父子行不能直接相加。Profiler 本身会扰动耗时，"
              "本例用单独的未插桩计时比较收益。",
              "", "此处只采集 CPU activity：GPU 运行时表格反映主机调度和等待，不是 GPU kernel 时间。"
              "wait_complete 可能包含等待 GPU 的时间，不能把它简单当成多余开销删掉。",
              "", "`*-trace.json` 可用兼容 Chrome Trace 的本地查看器查看；通常先读文本表即可。"
              "trace 可能包含进程等本地元数据，整个 results 目录保持 Git 忽略，不上传。",
              "", "练习：指出一个原始瓶颈假设、对应调用次数证据，以及优化后的数值正确性和耗时证据。"
              "这里模型没有训练，测试数据是随机像素；它验证预处理流程的性能，不验证模型质量。"]
    save(path, {"config": vars(a), "device": device, "timings": timings, "stages": profiles}, lines)


if __name__ == "__main__":
    main()
