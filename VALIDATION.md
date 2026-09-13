# 工程验证记录

这些记录表示代码路径已经试跑，不表示学习者已完成对应章节。

验证依赖：Python 3.11、PyTorch 2.14.0、Torchvision 0.29.0；完整依赖见 requirements-tested.txt。CPU 与 Apple MPS 已实测；CUDA 与 Windows 路径尚未实机验证。

| 检查 | 命令 | 结果 |
|---|---|---|
| Tensor 与梯度 | `python run.py 00` | 形状输出正确；一次更新后均方误差从约 18.667 降到 0.083 |
| CPU 快速训练 | `python run.py 01 --quick --device cpu` | 2 轮完成，验证准确率从约 11.3% 到 62.9%；保存/加载一致性通过 |
| MPS 默认训练 | `python run.py 01 --device mps` | 3 轮完成，验证准确率从约 12.4% 到 80.7%；保存/加载一致性通过 |
| 跨设备预测 | `python run.py predict --device cpu` | 在 CPU 加载 MPS 训练权重并输出预测，样本 PNG 正常保存 |
| MPS 预测 | `python run.py predict --device mps --index 15` | 加载、前向与样本导出通过 |
| CPU 小规模性能实验 | `python run.py 04 --device cpu --width 256 --repeats 3` | 正确性检查、原始样本和报告生成通过 |
| CPU/MPS 默认性能实验 | `python run.py 04 --device both` | 15 组记录完成，矩阵结果容差检查与报告输出通过 |

训练数据使用固定随机划分；上述验证准确率仅为教学配置的一次运行结果，不是官方完整测试集准确率或性能承诺。不同后端、依赖和运行条件会带来差异。

第一章首次下载使用 HTTPS 和 certifi 根证书，保留证书校验；数据由 torchvision 核验完整性。数据、权重和包含本地环境信息的原始输出均未纳入 Git。
