import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
import matplotlib.pyplot as plt

d2l.use_svg_display()

trans = transforms.ToTensor()

#1加载数据集
train_mnist = torchvision.datasets.FashionMNIST(
    root='../data', train=True, transform=trans, download=True  
)
#print((train_mnist[0][0].shape))
#print(train_mnist.classes)

#2创建dataloader
'''X, y = next(iter(data.DataLoader(train_mnist, batch_size=18, )))

d2l.show_images(X.reshape(18, 28, 28), 2, 9, titles=d2l.get_fashion_mnist_labels(y))
plt.show()'''

batch_size = 256 
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize = 64)
for X, y in test_iter:
    print(X.shape, X.dtype, y.shape, y.dtype)
    break

