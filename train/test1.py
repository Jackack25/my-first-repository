import torch 

b = torch.zeros(10, requires_grad=True)
print(b)

from functools import reduce
def fn(a, b):
    return 10 * a + b
print(reduce(fn, [1, 2, 3, 4, 5]))