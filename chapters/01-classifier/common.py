import os
from pathlib import Path

import certifi
import torch
from torch import nn
from torch.utils.data import Subset
from torchvision.datasets import FashionMNIST
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
ARTIFACTS = ROOT / "artifacts" / "fashion"
CLASSES = ["T恤", "裤子", "套头衫", "连衣裙", "外套", "凉鞋", "衬衫", "运动鞋", "包", "短靴"]


class FashionData(FashionMNIST):
    # 使用数据集作者的 HTTPS 镜像；仍由 torchvision 核验文件 MD5。
    mirrors = ["https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/"]


def dataset(train):
    # 部分 Python 安装缺少系统 CA 路径；提供受信任根证书，保持 HTTPS 校验开启。
    # 如果用户已经指定证书配置，则保留该配置。
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    return FashionData(root=DATA, train=train, transform=ToTensor(), download=True)


def split_data(train_size, val_size):
    full = dataset(train=True)
    order = torch.randperm(len(full), generator=torch.Generator().manual_seed(42)).tolist()
    # 最后 5000 个位置固定属于验证池，与训练池不重叠；不同规模实验保持划分边界。
    return Subset(full, order[:train_size]), Subset(full, order[-5000:][:val_size])


def build_model():
    return nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, 128), nn.ReLU(), nn.Linear(128, 10))


def choose_device(name):
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else (
            "mps" if torch.backends.mps.is_available() else "cpu")
    if name == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS 不可用，请在支持的 Mac 本机运行或使用 --device cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA 不可用，请检查构建/驱动或使用 --device cpu")
    return name


def sync(device):
    if device == "mps":
        torch.mps.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()
