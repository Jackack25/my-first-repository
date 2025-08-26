from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.optim 


#step 1 load data
class MyDataset(Dataset):
    def __init__(self, file):
        self.data = ...

dataset = MyDataset(file)
dataloader = DataLoader(dataset, batch_size, shuffle)


#step 2 define nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 64)
            nn.ReLu()
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = ...
        return self.net(x)
    
#step 3 LOSS function

#step 4 optimizer

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


#step 5 train loop
for epoch in range(10):
    model.train()
    for x, y in dataloader:
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

#step 6 validation loop
model.eval()
with torch.no_grad():
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)

#step 7 testing loop