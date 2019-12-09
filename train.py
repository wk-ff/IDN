import torch

from dataset.dataset import dataset
from models.net import net
from loss import loss

# cuda = torch.cuda.is_available()
cuda = False
train_set = dataset(train=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)

model = net()
if cuda:
    model = model.cuda()
criterion = loss()
if cuda:
    criterion = criterion.cuda()

for i, data in enumerate(train_loader):
    inputs, labels = data
    # label = torch.tensor(labels)
    # print(inputs.dtype, type(labels))

    if cuda:
        inputs, labels = inputs.cuda(), labels.cuda()
    
    predicted = model(inputs)

