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

## 新增章节验证

第 02、03、05 章已在 CPU 和 Apple MPS 上实际运行，不新增第三方依赖。CUDA 与 Windows 尚未实机验证。

| 检查 | 配置 | 结果 |
|---|---|---|
| 02 学习率对照 | CPU，默认三组学习率、5 轮 | 三组 epoch=0 指标一致，逐轮数据与报告完整；lr=0.001 本次验证准确率约 74.6% |
| 02 随机标签 | CPU，128 个训练样本、40 轮、lr=0.001 | 训练准确率约 90.6%，真实验证准确率约 4.7%，展示拟合训练标签不代表泛化 |
| 02 GPU 路径 | MPS，3 轮、lr=0.001 | 训练、验证及报告生成通过 |
| 03 批量与设备 | 默认 512 样本、4 个 batch、两种输入模式 | CPU/MPS 输出均与同一 CPU 参考结果容差一致，原始计时与报告生成通过 |
| 03 尾批 | CPU，513 样本、batch=64 | 9 次前向调用，输出完整且正确 |
| 05 Profiler | CPU/MPS，默认 batch=128 | 两种预处理输出一致；四个命名阶段、算子表与 trace 均生成 |

Profiler 的加速比以独立的未插桩计时计算；默认 CPU 示例本次约 2.68×，不是固定性能保证。主机侧事件不能当作 GPU kernel 时间。新章节生成的结果与 trace 均只保留本机。
