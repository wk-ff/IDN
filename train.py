import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pynvml import *
import os
from tensorboardX import SummaryWriter
import time
from tqdm import tqdm

from dataset.dataset import dataset
from models.net import net
from loss import loss
# os.environ['CUDA_VISIBLE_DEVICES']='1'

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


def compute_accuracy(predicted, labels):
    for i in range(3):
        predicted[i][predicted[i] > 0.5] = 1
        predicted[i][predicted[i] <= 0.5] = 0
    predicted = predicted[0] + predicted[1] + predicted[2]

    predicted[predicted < 2] = 0
    predicted[predicted >= 2] = 1
    predicted = predicted.view(-1)
    accuracy = torch.sum(predicted == labels).item() / labels.size()[0]
    return accuracy


BATCH_SIZE = 32
EPOCHS = 1
LEARNING_RATE = 0.001

np.random.seed(0)
torch.manual_seed(1)

train_set = dataset(train=True)
test_set = dataset(train=False)
train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=2*BATCH_SIZE, shuffle=False)

model = net().to(device)

criterion = loss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
# optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

writer = SummaryWriter(log_dir='scalar')

criterion = criterion.to(device)
iter_n = 0
t = time.strftime("%m-%d-%H-%M", time.localtime())
print(len(train_loader))
best_test_accuracy = 0
for epoch in tqdm(range(1, EPOCHS + 1)):
    for i, (inputs, labels) in enumerate(tqdm(train_loader)):
        torch.cuda.empty_cache()

        optimizer.zero_grad()

        labels = labels.float()
        inputs, labels = inputs.to(device), labels.to(device)

        predicted = model(inputs)

        loss = criterion(*predicted, labels)

        loss.backward()
        optimizer.step()

        accuracy = compute_accuracy(predicted, labels)

        writer.add_scalar(t+'/train_loss', loss.item(), iter_n)
        writer.add_scalar(t+'/train_accuracy', accuracy, iter_n)
        print(f'loss: {loss.item()}, accuracy: {accuracy}')

        if i % 100 == 0:
            with torch.no_grad():
                accuracys = []
                for i_, (inputs_, labels_) in enumerate(test_loader):
                    labels_ = labels_.float()
                    inputs_, labels_ = inputs_.to(device), labels_.to(device)
                    predicted_ = model(inputs_)
                    accuracys.append(compute_accuracy(predicted_, labels_))
                accuracy_ = sum(accuracys) / len(accuracys)
                writer.add_scalar(t+'/test_accuracy', accuracy_, iter_n)
            print(f'test loss:{accuracy_:.6f}')
            if accuracy_ >= best_test_accuracy:
                best_test_accuracy = accuracy_
                torch.save(model.state_dict(), f'model_{accuracy_:%}.pth')

        iter_n += 1

        print('Epoch[{}/{}], iter {}, loss:{:.6f}, accuracy:{}'.format(epoch,
              EPOCHS, i, loss.item(), accuracy))

writer.close()
