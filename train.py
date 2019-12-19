import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynvml import *
import os

from dataset.dataset import dataset
from models.net import net
from loss import loss
os.environ['CUDA_VISIBLE_DEVICES']='2'

# nvmlInit()
# handle = nvmlDeviceGetHandleByIndex(0)
# info = nvmlDeviceGetMemoryInfo(handle)

def compute_accuracy(net, test_loader):
    net.eval()
    with torch.no_grad():
        accuracys = []
        for i, (inputs, labels) in enumerate(test_loader):
            labels = labels.float()
            if torch.cuda.is_available():
                inputs, labels = inputs.cuda(), labels.cuda()
            predicted = net(inputs)

            for i in range(3):
                predicted[i][predicted[i] > 0.5] = 1
                predicted[i][predicted[i] <= 0.5] = 0
            predicted = predicted[0] + predicted[1] + predicted[2]
            predicted = torch.sum(predicted, dim=1)
            predicted[predicted >= 2] = 1
            predicted[predicted < 2] = 0
            
            accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
            accuracys.append(accuracy)
        
        return sum(accuracys) / len(accuracys)


BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

np.random.seed(0)
torch.manual_seed(1)

cuda = torch.cuda.is_available()

train_set = dataset(train=False)
test_set = dataset(train=False)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

model = net()
if cuda:
    model = model.cuda()
criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

if cuda:
    criterion = criterion.cuda()
for epoch in range(1, EPOCHS + 1):
    for i, (inputs, labels) in enumerate(train_loader):
        torch.cuda.empty_cache()
        # modelsize(model, inputs)
        # inputs, labels = data
        # print(i, inputs.size())
        optimizer.zero_grad()
        
        # label = torch.tensor(labels)
        # print(inputs.dtype, type(labels))

        labels = labels.float()
        if cuda:
            inputs, labels = inputs.cuda(), labels.cuda()
        
        # print("Memory Total:{}, Free:{}, Used:{}".format(info.total, info.free, info.used))
        # print(labels.size())
        # with torch.no_grad():
        predicted = model(inputs)
        if torch.max(predicted[0]).item() > 1 or torch.min(predicted[0]).item()<0:
            print(torch.max(predicted[0]).item())
        # print(predicted[0].size())

        loss = criterion(*predicted, labels)
        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(model, test_loader)

        print('accuracy:{}'.format(accuracy))

        # if epoch < 2:
        #     lr = lr * 2
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr
        # else:
        #     lr = lr * 0.1
        #     for param_group in optimizer.param_groups:
        #         param_group['lr'] = lr

        if i % 1 == 0:
            print('Epoch[{}/{}], iter {}, loss: {}'.format(epoch, EPOCHS, i, loss.item()))
        # print("Memory Total:{}, Free:{}, Used:{}".format(info.total, info.free, info.used))

