import torch
from d2l import torch as d2l
import matplotlib.pyplot as plt

batch_size = 256
num_inputs = 28*28
num_outputs = 10
lr = 0.1
num_epochs = 10

# 1.定义参数,更新的参数
W = torch.normal(0, 0.01, size = (num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)
#print(W.shape[0])
                
# 2.加载dataset dataloader
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)



# 3.定义softmax函数
def softmax(X):
    # 1.首先指数运算
    X_exp = torch.exp(X)
    # 2.然后除以所有指数的和
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp/ partition

# 4.定义模型函数
def net(X):
    # X:(batch_size,channel, H, W)
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b)


# 5.定义交叉熵损失函数
def cross_entropy(y_hat, y):
    return -torch.log(y_hat[range(len(y_hat)),y])
    

# 6.定义我们的优化器
def updater(bacth_size):
    d2l.sgd([W, b], lr, bacth_size)

# 7训练模型和测试



if __name__ == '__main__':
    '''X = torch.linspace(1, 6, 6).view(2, 3)
    print(X)
    print(X.sum(0, keepdim=True))
    print(X.sum(1, keepdim=True))
    x_prob = softmax(X)
    print(x_prob)
    print(x_prob.sum(1))
'''
    
    # y = torch.tensor([0, 2])
    # y_hat = torch.tensor([[0.1, 0.2, 0.7], [0.2, 0.3, 0.5]])
    d2l.train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
    d2l.predict_ch3(net, test_iter)
    plt.show()