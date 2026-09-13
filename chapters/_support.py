"""复用第 01 章的模型/数据与设备选择，避免章节间复制核心逻辑。"""
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "chapters" / "01-classifier"))
from common import build_model, choose_device, split_data, sync


def output_dir(chapter):
    path = ROOT / "results" / chapter / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    path.mkdir(parents=True, exist_ok=False)
    return path


def save(path, data, lines):
    (path / "results.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (path / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"中文报告：{(path / 'report.md').relative_to(ROOT)}", flush=True)


def timed(fn, device):
    sync(device)
    start = time.perf_counter_ns()
    fn()
    sync(device)
    return (time.perf_counter_ns() - start) / 1e6


def measure_variants(variants, device, repeats):
    # 同一设备上交替先后顺序；每种方式先预热 3 次。
    for fn in variants.values():
        for _ in range(3):
            fn()
        sync(device)
    samples = {name: [] for name in variants}
    names = list(variants)
    for i in range(repeats):
        for name in names[::1 if i % 2 == 0 else -1]:
            samples[name].append(timed(variants[name], device))
    return {name: {"median_ms": statistics.median(values), "samples_ms": values}
            for name, values in samples.items()}


@torch.inference_mode()
def classification_metrics(model, x, y):
    model.eval()
    logits = model(x)
    return {"loss": torch.nn.functional.cross_entropy(logits, y).item(),
            "accuracy": (logits.argmax(1) == y).float().mean().item()}
