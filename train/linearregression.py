import torch

device = torch.devcie('cuda' if torch.cuda.is_availavle() else 'cpu')

#生成数据集
inputs = torch.rand(100, 3)
weights = torch.tensor([[1.1], [2.2], [3.3]])
bias = torch.tensor([4.4])
targets = inputs @ weights + bias + 0.1 * torch.randn(100, 1)

#初始化参数,并开启梯度追踪
w = torch.ones((3, 1), requires_grad=True, device = device)
b = torch.ones(1, requires_grad=True, device = device)

#训练
#1.将数据移动至相同设备
# 将数据移至相同设备
inputs = inputs.to(device)
targets = targets.to(device)

#设置超参数
epoch = 10000
lr = 0.003

for i in range(epoch):
    outputs = inputs @ w + b
    loss = torch.mean(torch.square(outputs - targets))
    print("loss:", loss.item())

    loss.backward()

    with torch.no_grad(): #下边的计算不需要跟踪梯度
        w -= lr * w.grad
        b -= lr * b.grad

    # 清零梯度
    w.grad.zero_()
    b.grad.zero_()

print("训练后的权重 w:", w)
print("训练后的偏置 b:", b)