# AI Infra Learning

从一个 Tensor 和一次参数更新开始，通过小 demo 逐步理解模型训练、推理和性能问题。每章都有中文说明、代码、练习和面试问题。

## 章节

| 章节 | 学习问题 | 状态 |
|---|---|---|
| [00 · Tensor 与一次学习](chapters/00-tensors/README.md) | 数据是什么形状？梯度如何改变参数？ | 可运行，建议从这里开始 |
| [01 · 第一个分类模型](chapters/01-classifier/README.md) | 一批图片如何完成训练、验证、保存与预测？ | 可运行 |
| [02 · 理解训练结果](chapters/02-training-results/README.md) | 学习率、loss、准确率与过拟合如何联系？ | 可运行 |
| [03 · batch 与设备](chapters/03-batches-devices/README.md) | 同一批样本，推理批量与输入搬运如何影响速度？ | 可运行 |
| [04 · 性能小实验](chapters/04-performance/README.md) | 形状、批处理、异步执行如何影响计时？ | 可运行，建议稍后学习 |
| [05 · Profiler 排查](chapters/05-profiler/README.md) | 如何用调用次数和独立计时验证一次优化？ | 可运行 |

后续路线见 [ROADMAP.md](ROADMAP.md)。学习进度和跨电脑接续位置见 [PROGRESS.md](PROGRESS.md)。

## 安装一次

在仓库根目录运行，建议 Python 3.11 或 3.12：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Windows PowerShell 使用 `py -3 -m venv .venv`，然后 `.venv\Scripts\Activate.ps1`；后续 Python 命令相同。

`requirements-tested.txt` 记录首轮验证版本，日常安装使用 `requirements.txt`。不同平台的 GPU 支持取决于 PyTorch 构建和驱动；没有 GPU 也能完成前两章。

## 今天只做第 00 章

```bash
python run.py 00
```

运行后解释三件事：`shape` 表示什么、矩阵乘法为什么得到这个形状、为什么更新后误差变小。再看[第 00 章的练习](chapters/00-tensors/README.md)。

后续命令：

```bash
python run.py 01 --quick
python run.py predict
python run.py 02
python run.py 03
python run.py 04
python run.py 05
```

第 00、03、04、05 章不需要联网；第 01、02 章复用公开 Fashion-MNIST 数据集，首次使用会下载。生成数据、模型、报告和 Profiler trace 只保存在本地。没有任何云服务密钥要求。

建议按 **00 → 01 → 02 → 03 → 04 → 05** 顺序，每次只运行一章：先读问题并预测，运行后看终端输出或生成的 `report.md`，再改一个变量完成练习。新章节不要求你已训练出可用权重。

## 两台电脑如何接着学

1. 另一台电脑登录有权限的 GitHub 账号，克隆此仓库，按上面的命令安装依赖。
2. 在 Codex 中打开克隆后的目录，发送：“读取 AGENTS.md 和 PROGRESS.md，从当前章节继续带我学习。”
3. 换电脑前更新 PROGRESS.md，把本次理解、问题与下一步写进去，再检查、提交并推送。
4. 另一台电脑先 `git pull --ff-only`。如果有本地修改或两边都提交过，先检查差异并解决冲突，不使用强制推送覆盖。

同步的是教学资料、代码和显式记录的进度；本地模型和数据需要重新生成。仓库不保存聊天原文。

新克隆需要设置一次通用提交身份，以免继承全局真实姓名和邮箱：

```bash
git config --local user.name 'AI Infra Learner'
git config --local user.email 'learner@users.noreply.github.com'
```

提交时逐个选择需要分享的文件：

```bash
git status --short
git add PROGRESS.md
python tools/check_share.py
git diff --cached
git commit -m "Record learning progress"
git push
```

代码变动也需要显式 `git add` 对应文件。检查器只是辅助检查，不能保证识别所有个人信息；提交前仍要阅读差异。分享范围见 [PRIVACY.md](PRIVACY.md)。
