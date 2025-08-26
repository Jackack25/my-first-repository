import torch

x = torch.ones([2, 2, 1])
y = torch.ones([2, 2, 2])
z = torch.ones([2, 2, 3])
print
print(x)
print(y)
print(z)
w = torch.cat((x, y, z), dim=2)
print(w)
print(w.shape)

a = torch.tensor([1, 2, 3], [4, 5, 6], requires_grad=True)
