"""重新加载本地权重，对公开测试集的一张图片预测。"""
import argparse
import json
from pathlib import Path

import torch

from common import ARTIFACTS, CLASSES, ROOT, build_model, choose_device, dataset


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--index", type=int, default=0)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    a = p.parse_args()
    try:
        device = choose_device(a.device)
    except ValueError as e:
        p.error(str(e))
    checkpoint = a.checkpoint
    if checkpoint is None:
        latest = ARTIFACTS / "latest.json"
        if not latest.exists():
            p.error("尚无训练结果，请先运行 python run.py 01 --quick")
        checkpoint = ARTIFACTS / json.loads(latest.read_text())["run"] / "model.pt"
    if not checkpoint.is_file():
        p.error("指定的模型文件不存在")
    test_data = dataset(train=False)
    if not 0 <= a.index < len(test_data):
        p.error(f"index 需为 0–{len(test_data)-1}")
    model = build_model()
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.to(device).eval()
    image, label = test_data[a.index]
    with torch.inference_mode():
        probs = model(image.unsqueeze(0).to(device)).softmax(dim=1)[0].cpu()
    values, indices = probs.topk(3)
    output = checkpoint.parent / f"sample-{a.index}.png"
    from torchvision.transforms.functional import to_pil_image
    to_pil_image(image).resize((280, 280)).save(output)
    print(f"样本={a.index}；真实类别={CLASSES[label]}；设备={device}")
    for score, index in zip(values.tolist(), indices.tolist()):
        print(f"  {CLASSES[index]}：softmax 分数 {score:.1%}")
    print("这些分数不是经过校准的正确概率；单张预测也不代表整体准确率。")
    print(f"图片已保存：{output.relative_to(ROOT) if output.is_relative_to(ROOT) else output.name}")


if __name__ == "__main__":
    main()
