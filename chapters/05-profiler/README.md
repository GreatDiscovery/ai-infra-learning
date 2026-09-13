# 05 · Profiler：第一次有证据的优化

先完成第 03–04 章，知道计时边界与同步的作用。本章从一个可以复现的小问题开始：逐张做预处理会有什么成本？

## 先运行 CPU 版本

```bash
python run.py 05
```

无需下载模型或数据。程序生成一个 batch 的随机灰度像素，并建立与第 01 章同结构的未训练分类器。

两种实现完成相同工作：

```python
# 逐张转换再拼起来
images = torch.stack([image.float() / 255.0 for image in raw])

# 一次转换整个 batch
images = raw.float() / 255.0
```

程序先检查两种实现产生的模型输出一致，再单独计时，最后采集 Profiler 数据。这里没有通过 sleep 人为制造瓶颈。

## 输出去哪了

`results/05-profiler/时间戳/` 包含：

- `report.md`：未插桩耗时对比、主机侧阶段耗时及中文解读。
- `results.json`：原始计时、参数与阶段数据。
- `row_loop-operators.txt` / `vectorized-operators.txt`：算子耗时与调用次数表。
- `*-trace.json`：时间线，可用支持 Chrome Trace 的本地查看器查看；初学时读文本表就够了。

trace 可能含进程标识等本地元数据，始终只留本机；不要将整个 results 目录上传 GitHub。

## 四个阶段

1. `01_prepare_input`：在 CPU 上把整数像素变为 `[0,1]` 的浮点数。
2. `02_to_device`：把输入送到目标设备；CPU 同类型输入可能无需复制。
3. `03_forward`：执行模型。
4. `04_wait_complete`：等待 GPU 完成（CPU 模式为空操作）。

`record_function` 给代码区间起名，便于在 Profiler 中找到它。它本身不优化代码。

先找 prepare_input 的耗时和 `aten::div` 等算子的调用次数，比较逐张处理与整体处理：相同计算可以因为调用方式不同而花费不同时间。默认配置通常能观察到这种差别，实际收益以本次测量为准。

## 两个容易误读的地方

**Self CPU 与 CPU total**：前者排除子操作，后者包括子操作。如果把父区间和内部算子耗时都相加，会重复计数。

**Profiler 耗时与正常耗时**：记录事件本身有成本，尤其会影响大量小算子。因此收益对比使用单独的未开启 Profiler 的运行，预热后交替测量两种实现。

本例只采集 `ProfilerActivity.CPU`。即使选择 MPS/CUDA，看到的也是主机调用和等待；不能把 forward 的 CPU 时间当作 GPU kernel 时间。第 04 阶段可能是在等待已经提交的 GPU 工作，不能直接删掉等待然后宣称计算更快。

## 动手顺序

1. 跑默认版本，记录两种方式的耗时比。
2. 打开两份 operators 表，找到一个明显减少调用次数的算子，解释为何减少。
3. 改 batch size，预测这种优化对小/大 batch 的收益有何区别。

```bash
python run.py 05 --batch-size 8
python run.py 05 --batch-size 256
python run.py 05 --device mps
```

## 面试问答

<details><summary>你如何证明优化有效？</summary>

先保证输入、输出和工作量可比，做结果一致性检查；再使用相同条件的未插桩计时，重复运行；Profiler 用来解释调用和耗时变化。
</details>

<details><summary>最耗时的函数是不是一定能优化？</summary>

不一定。它可能包含必要的子工作或等待，需要先区分自身时间、子操作、同步和真实瓶颈。收益也受到可优化部分占总耗时比例的限制。
</details>

<details><summary>这次改动的代价或适用条件是什么？</summary>

整体转换要求当前 batch 已经能组成一个张量；如果真实数据尺寸不同或逐样本变换不同，不能直接套用。还应考虑临时内存和实际流水线结构。
</details>

完成标准：能讲清“现象 → 假设 → 调用次数证据 → 正确性 → 独立计时收益”。把这个过程用自己的话记到 PROGRESS.md。

参考：[PyTorch Profiler 教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)。
