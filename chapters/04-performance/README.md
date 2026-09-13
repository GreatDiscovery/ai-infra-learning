# 04 · 性能小实验

建议完成前两章后再学习。三个实验分别改变矩阵形状、逐行/批量调用方式和 GPU 计时边界。

```bash
python run.py 04
```

默认 K=N=1024，M 为 1、8、32、128、512。自动对比 CPU 与可用 GPU（优先 CUDA，其次 Apple MPS）；没有 GPU 就运行前两个 CPU 实验。

每组预热 5 次、重复测量 15 次，包含结果正确性检查。输出在 `results/performance/时间戳/`：中文 `report.md`、原始样本 `results.json`、汇总 `results.csv`。这些结果仅留本地。

## 先猜再运行

1. M 从 1 到 512，计算量增加 512 倍，耗时也会吗？
2. 同样处理 64 行，64 次单行矩阵乘法与一次批量矩阵乘法，耗时差多少？
3. 仅计时到 GPU 调用返回，与等待计算完成后再停表，会相同吗？

```bash
python run.py 04 --width 2048
python run.py 04 --device cpu --threads 1
python run.py 04 --device cpu --threads 4
python run.py 04 --device mps
python run.py 04 --device cuda
```

每次只改变一个变量。第 01 章训练计时包含更多步骤，这里聚焦已在设备上的张量计算，使用预分配输出，不包含数据生成与搬运。

## 如何解释

计算量近似 `2*M*K*N` FLOPs，吞吐为计算量除以耗时。任务变大时，耗时可能增加，同时吞吐也提高。

算术强度估算为 `2*M*K*N / (4*(M*K + K*N + M*N))`，单位 FLOP/byte；它假设 FP32 输入各读一次、输出写一次。缓存和分块会改变实际流量，所以这个估算不等于显存带宽测量，也不能独自证明 memory-bound（主要受数据搬运限制）。

逐行与批量比较同时改变调用次数、矩阵形状与数据复用；收益不能全部归因于 Python 循环。真实服务还需要考虑凑批等待与排队延迟，本例未模拟。

GPU 异步调用可以先返回，代码会在计时前后同步等待。测量结果包含主机调度、调用与同步成本，不是独立 kernel 时间。小任务 GPU 比 CPU 慢并不异常；须结合任务规模、后端和这些开销解释。

## 面试练习

- 为什么性能测试需要预热？首次内核/运行时准备和缓存状态可能使首次执行不同于后续执行。
- 为什么需要同步？要明确计时到提交完成，还是计算完成；异步调用返回不能作为完成的证据。
- 批量越大越好吗？吞吐可能改善，但服务等待和内存占用也可能增加；先说明优化目标。
- 如何证明瓶颈？本例是提出假设的起点，后续结合 profiler、对照实验与硬件能力验证，不能只看 GPU 利用率下结论。

本章没有实现完整 Transformer、KV Cache、Prefill/Decode 或线上服务。矩阵实验仅用于理解其中部分计算的性能因素。CUDA 路径需要在对应硬件上独立验证。

参考：[PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)、[CUDA 同步接口](https://docs.pytorch.org/docs/stable/generated/torch.cuda.synchronize.html)、[矩阵乘法性能基础](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)。
