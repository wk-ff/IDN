import argparse
import torch
from dataset.dataset import dataset
from models.net import net


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='', type=str,
                        help='directory of testing model')
    args = parser.parse_args()
    return args


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


if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

BATCH_SIZE = 32
test_set = dataset(train=False)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=2*BATCH_SIZE, shuffle=False)
args = parse_args()
assert args.model_dir != '', 'model_dir is required'

model = net().to(device)
model.load_state_dict(torch.load(args.model_dir))
with torch.no_grad():
    accuracys = []
    for i_, (inputs_, labels_) in enumerate(test_loader):
        labels_ = labels_.float()
        inputs_, labels_ = inputs_.to(device), labels_.to(device)
        predicted_ = model(inputs_)
        accuracys.append(compute_accuracy(predicted_, labels_))
    accuracy_ = sum(accuracys) / len(accuracys)
print(f'test accuracy:{accuracy_:%}')
