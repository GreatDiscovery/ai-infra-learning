"""控制其他条件，只改变学习率，观察完整训练/验证曲线。"""
import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _support import build_model, choose_device, classification_metrics, output_dir, save, split_data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train-size", type=int, default=1024)
    p.add_argument("--val-size", type=int, default=512)
    p.add_argument("--lrs", type=float, nargs="+", default=[0.00001, 0.001, 0.1])
    p.add_argument("--random-labels", action="store_true", help="仅将训练标签改为固定随机标签")
    a = p.parse_args()
    if not (1 <= a.epochs <= 100 and 64 <= a.train_size <= 4096 and 64 <= a.val_size <= 2048):
        p.error("epochs 需为 1–100，train-size 需为 64–4096，val-size 需为 64–2048")
    if not 1 <= len(a.lrs) <= 5 or any(not math.isfinite(lr) or not 0 < lr <= 1 for lr in a.lrs):
        p.error("指定 1–5 个有限学习率，每个需在 (0, 1] 内")
    try:
        device = choose_device(a.device)
    except ValueError as e:
        p.error(str(e))
    torch.set_num_threads(4)
    training, validation = split_data(a.train_size, a.val_size)
    # 入门实验直接准备小规模张量；数据划分复用第 01 章，不使用官方测试集选参数。
    x = torch.stack([image for image, _ in training]).to(device)
    y = torch.tensor([label for _, label in training], device=device)
    vx = torch.stack([image for image, _ in validation]).to(device)
    vy = torch.tensor([label for _, label in validation], device=device)
    if a.random_labels:
        y = torch.randint(10, (len(y),), generator=torch.Generator().manual_seed(123)).to(device)
        print("注意：训练标签已替换为固定随机标签，验证标签保持真实。", flush=True)
    results = []
    print("先猜：学习率越大越好吗？train loss 下降，val loss 一定下降吗？")
    for lr in a.lrs:
        torch.manual_seed(42)  # 每个学习率使用相同初始化。
        model = build_model().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        order_generator = torch.Generator().manual_seed(7)  # 每轮相同的样本顺序。
        history, status = [], "completed"
        for epoch in range(a.epochs + 1):
            if epoch:
                model.train()
                indices = torch.randperm(len(y), generator=order_generator).to(device)
                for idx in indices.split(64):
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(model(x[idx]), y[idx])
                    if not torch.isfinite(loss).item():
                        status = "nonfinite_loss"
                        break
                    loss.backward()
                    optimizer.step()
            train = classification_metrics(model, x, y)
            val = classification_metrics(model, vx, vy)
            if not all(math.isfinite(v) for v in [*train.values(), *val.values()]):
                status = "nonfinite_metrics"
                break
            row = {"epoch": epoch, "train_loss": train["loss"], "train_accuracy": train["accuracy"],
                   "val_loss": val["loss"], "val_accuracy": val["accuracy"]}
            history.append(row)
            print(f"lr={lr:g} epoch={epoch:2d} | train loss={train['loss']:.3f} "
                  f"acc={train['accuracy']:.1%} | val loss={val['loss']:.3f} acc={val['accuracy']:.1%}", flush=True)
            if status != "completed":
                break
        results.append({"lr": lr, "status": status, "history": history})
    lines = ["# 02 · 训练结果", "", f"设备：{device}；参数：`{vars(a)}`。", "",
             "每轮结束后以同一份参数分别评估训练/验证数据；epoch=0 是未训练基线。", "",
             "| 学习率 | epoch | train loss | train acc | val loss | val acc |",
             "|---:|---:|---:|---:|---:|---:|"]
    for result in results:
        for r in result["history"]:
            lines.append(f"| {result['lr']:g} | {r['epoch']} | {r['train_loss']:.4f} | "
                         f"{r['train_accuracy']:.1%} | {r['val_loss']:.4f} | {r['val_accuracy']:.1%} |")
        if result["status"] != "completed":
            lines.append(f"\n学习率 {result['lr']:g} 提前停止：{result['status']}。\n")
    lines += ["", "## 先观察，再解释", "",
              "1. 对比每个学习率的 epoch=0：它们应相同，说明初始化和评估数据保持一致。",
              "2. 找一组 loss 下降而准确率不变或变化较小的相邻轮次；loss 还反映类别分数的变化。",
              "3. 训练表现持续改善而验证表现变差，才是可能过拟合的线索；一次波动不足以下结论。",
              "4. 默认运行不保证出现明显过拟合。随机训练标签实验可以检查模型是否会记忆无意义对应关系。",
              "", "这里改变的是 Adam 的学习率；结论不能直接迁移到其他优化器。验证集用于比较设置，"
              "不能把挑选后的验证表现当成最终测试结果。"]
    save(output_dir("02-training-results"), {"config": vars(a), "device": device, "results": results}, lines)


if __name__ == "__main__":
    main()
