import torch
from d2l import torch as d2l
from torch import nn


#1.生成数据
true_w = torch.tensor([2, 2.4])
true_b = 3.1
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
#print(features.shape, labels.shape)

#2.生成数据集,构造dataloader
batch_size = 10
dataloader = d2l.load_array((features, labels), batch_size)

#3.构造模型
net = nn.Linear(2, 1)
net.weight.data.normal_(0, 0.01)
net.bias.data.fill_(0)
print(net.weight.data, net.bias.data)

#4构造loss function
loss = nn.MSELoss()

#5.构造优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.03)


#6.迭代训练
for epoch in range(3):
    for X, y in dataloader:
        loss_result = loss(net(X), y)
        trainer.zero_grad()
        loss_result.backward()
        trainer.step()
    loss_check = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {loss_check:6f}')

print(f'真实w:{true_w},预测w:{net.weight.data}')
print(f'真实b:{true_b},预测b:{net.bias.data}')
