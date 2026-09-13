"""一条完整训练流程：数据 → 前向 → loss → 梯度 → 更新 → 验证 → 保存。"""
import argparse
import json
import time
from datetime import datetime

import torch
from torch import nn
from torch.utils.data import DataLoader

from common import ARTIFACTS, ROOT, build_model, choose_device, split_data, sync


@torch.inference_mode()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, correct, count = 0., 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return {"loss": loss_sum / count, "accuracy": correct / count}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--train-size", type=int, default=6000)
    p.add_argument("--val-size", type=int, default=1000)
    p.add_argument("--quick", action="store_true", help="使用 1024 训练样本、256 验证样本、2 轮")
    a = p.parse_args()
    if a.quick:
        a.train_size, a.val_size, a.epochs = 1024, 256, 2
    if not (1 <= a.train_size <= 55000 and 1 <= a.val_size <= 5000):
        p.error("train-size 需为 1–55000；val-size 需为 1–5000")
    if not (1 <= a.epochs <= 100 and 1 <= a.batch_size <= 4096 and 0 < a.lr <= 1):
        p.error("epochs 需为 1–100；batch-size 需为 1–4096；lr 需为 (0, 1]")
    try:
        device = choose_device(a.device)
    except ValueError as e:
        p.error(str(e))
    torch.manual_seed(42)
    torch.set_num_threads(4)
    train_data, val_data = split_data(a.train_size, a.val_size)
    train_loader = DataLoader(train_data, batch_size=a.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=a.batch_size, num_workers=0)
    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=a.lr)
    print(f"设备={device}；训练={len(train_data)}；验证={len(val_data)}；batch={a.batch_size}")
    print("输入 [B,1,28,28] → 展平 [B,784] → 隐藏层 [B,128] → 类别分数 [B,10]")
    baseline = evaluate(model, val_loader, device, criterion)
    print(f"训练前验证：loss={baseline['loss']:.4f}，准确率={baseline['accuracy']:.1%}", flush=True)
    history = []
    for epoch in range(1, a.epochs + 1):
        model.train()
        sync(device)
        start = time.perf_counter()
        loss_sum, count = 0., 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)  # 清理上一批留下的梯度。
            logits = model(images)                 # 前向：当前参数给出的类别分数。
            loss = criterion(logits, labels)        # 与真实标签比较；直接输入 logits。
            loss.backward()                        # 求梯度，此时尚未更新参数。
            optimizer.step()                       # 用梯度更新参数。
            loss_sum += loss.item() * len(labels)
            count += len(labels)
        sync(device)
        elapsed = time.perf_counter() - start
        val = evaluate(model, val_loader, device, criterion)
        row = {"epoch": epoch, "train_loss": loss_sum / count,
               "val_loss": val["loss"], "val_accuracy": val["accuracy"], "train_seconds": elapsed}
        history.append(row)
        print(f"第 {epoch} 轮：train loss={row['train_loss']:.4f}，"
              f"val loss={val['loss']:.4f}，val acc={val['accuracy']:.1%}，"
              f"训练用时={elapsed:.2f}s", flush=True)

    # 每次运行单独保存；不覆盖已有模型，latest.json 只指向最近一次结果。
    run_dir = ARTIFACTS / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    run_dir.mkdir(parents=True)
    checkpoint = run_dir / "model.pt"
    cpu_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(cpu_state, checkpoint)
    restored = build_model()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    original_cpu = build_model()
    original_cpu.load_state_dict(cpu_state)
    probe = val_data[0][0].unsqueeze(0)
    with torch.inference_mode():
        torch.testing.assert_close(restored(probe), original_cpu(probe), rtol=0, atol=0)
    metrics = {"config": vars(a), "device": device, "torch": torch.__version__,
               "baseline": baseline, "epochs": history, "checkpoint_roundtrip_passed": True}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (ARTIFACTS / "latest.json").write_text(json.dumps({"run": run_dir.name}), encoding="utf-8")
    print(f"保存并重新加载验证通过：{checkpoint.relative_to(ROOT)}")
    print("下一步：python run.py predict")


if __name__ == "__main__":
    main()
