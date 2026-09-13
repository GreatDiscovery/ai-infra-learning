# 01 · 第一个分类模型

目标：完整走通读数据、训练、验证、保存、重新加载和单张预测。先完成第 00 章。

## 运行

```bash
python run.py 01 --quick
python run.py predict
```

第一次会下载公开 Fashion-MNIST 数据集，保存在 `data/`。快速模式使用 1024 个训练样本、256 个验证样本、2 轮训练，适合先确认流程；不要期待它达到完整训练的准确率。

每次训练会在 `artifacts/fashion/时间戳/` 保存 `model.pt` 与 `metrics.json`；预测会保存所选公开样本的放大 PNG，可以对照图片、真实标签和模型输出。`latest.json` 指向最近一轮，不覆盖旧模型。

稍后再运行默认配置（6000 个训练样本、1000 个验证样本、3 轮）：

```bash
python run.py 01
python run.py predict --index 15
```

## 沿一批图片读代码

图片是 28×28 灰度图。`ToTensor()` 把像素转换为 `[0,1]` 的浮点数，一批图片形状是 `[B,1,28,28]`。

模型的路径：`[B,1,28,28] → [B,784] → [B,128] → [B,10]`。最后的 10 个数字是类别分数（logits），并非已经归一化的概率。

请从 `train.py` 内层循环的这五行读起：

```python
optimizer.zero_grad(set_to_none=True)
logits = model(images)
loss = criterion(logits, labels)
loss.backward()
optimizer.step()
```

`CrossEntropyLoss` 接受原始类别分数和整数标签，不用先做 softmax。`backward()` 算梯度；`step()` 更新参数。PyTorch 默认会累积梯度，因此每批先清理上一批的梯度。

## 验证与保存

从官方训练数据中固定随机划出互不重叠的训练池与验证池；官方测试集不参与训练或选参数，只在预测示例中查看个别样本。调整训练规模时不改变验证池边界。

每轮先训练，再验证。训练模式 `model.train()` 和评估模式 `model.eval()` 控制部分层的行为；`inference_mode()` 关闭求导相关开销，二者用途不同。本章的简单模型没有 Dropout/BatchNorm，但保留这套标准用法。

保存的是参数 `state_dict`。加载时重新建立相同结构，并用 `weights_only=True` 读取本地权重。程序会核对保存/加载前后在同一 CPU 输入上的输出。

报告中的训练时间包括数据读取/转换、搬运、计算和 Python 开销，不含数据下载与验证；每批 `loss.item()` 也可能产生同步。它用于观察流程，不是纯 GPU 性能测试。

## 练习：每次只做一个

1. 看输出，对照代码指出前向、求梯度和更新参数分别在哪一行。
2. 默认配置运行 1 轮与 3 轮，记录验证准确率。命令用 `--epochs 1` / `--epochs 3`，不要加会覆盖轮数的 `--quick`。
3. 改 `--batch-size 16` 和 `--batch-size 128`，记录每轮迭代次数与时间。此时同样轮数下参数更新次数也改变了，不要把准确率差异只归因于硬件。
4. 使用 `--device cpu` 与 `--device mps` 比较；有 NVIDIA 环境时可用 `--device cuda`。先预测，记录条件，再观察。

## 面试练习

<details><summary>batch、iteration、epoch 有什么区别？</summary>

batch 是一批样本；这里一次 iteration 处理一批并更新一次参数；epoch 是遍历训练集一轮。没有丢弃最后一批时，每轮迭代数为训练样本数除以 batch size 后向上取整。
</details>

<details><summary>为什么不只看训练准确率？</summary>

训练数据参与了参数学习，在它上面表现好不代表对未见样本也好。验证集用于比较设置；测试集应留到最终评估，不能反复用它选参数。
</details>

<details><summary>eval() 与禁用梯度是同一回事吗？</summary>

不是。eval() 切换部分层的行为，inference_mode()/no_grad() 控制梯度记录。通常推理时两者一起使用。
</details>

完成标准：亲自训练、找到权重文件、重新加载预测，并用自己的话解释训练循环。先把理解和疑问写到 PROGRESS.md；性能优化可稍后再学。

参考：[Fashion-MNIST 官方项目](https://github.com/zalandoresearch/fashion-mnist)、[PyTorch 训练入门](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)、[保存与加载](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)。数据不随仓库分发，请参考原项目许可。
