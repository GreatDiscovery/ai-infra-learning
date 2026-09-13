"""先理解数据与一次参数更新，无需数据下载和 GPU。"""
import torch


def main():
    print("1. 一个 batch = 一次处理的一组样本")
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    w = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
    print("x =", x)
    print("x.shape =", tuple(x.shape), "→ 2 个样本，每个 3 个特征")
    print("w.shape =", tuple(w.shape), "→ 把 3 个特征映射成 2 个输出")
    print("x @ w =", x @ w)
    print("输出形状：", tuple((x @ w).shape))

    print("\n2. 最小模型：预测值 = 参数 × 输入")
    inputs = torch.tensor([1., 2., 3.])
    targets = torch.tensor([2., 4., 6.])
    weight = torch.tensor(0., requires_grad=True)
    prediction = weight * inputs
    loss = ((prediction - targets) ** 2).mean()
    print(f"更新前：参数={weight.item():.3f}，均方误差={loss.item():.3f}")
    loss.backward()  # 计算误差对参数的导数；这一步还没有修改参数。
    print(f"梯度={weight.grad.item():.3f}：当前位置增加参数会让误差下降")
    learning_rate = 0.1
    with torch.no_grad():  # 更新参数本身不需要建立求导计算图。
        weight -= learning_rate * weight.grad
        new_loss = ((weight * inputs - targets) ** 2).mean()
    print(f"更新后：参数={weight.item():.3f}，均方误差={new_loss.item():.3f}")
    assert new_loss < loss, "本示例的学习率应该使一次更新后的误差下降"
    print("\n练习：把 learning_rate 改成 1.0。先猜误差如何变化，再运行。")
    print("如果断言失败，说明较大的这一步使误差上升；解释原因后恢复参数。")


if __name__ == "__main__":
    main()
