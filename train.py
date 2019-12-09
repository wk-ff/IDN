import torch

from dataset.dataset import dataset
from models.net import net

train_set = dataset(train=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)

for i, data in enumerate(train_loader):
    inputs, labels = data
    print(inputs.size())

