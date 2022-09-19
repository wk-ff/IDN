import argparse
from re import L
import torch
from dataset.dataset import SignatureLoader
from models.net import net
from utils import *
from sklearn import metrics
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', default='', type=str,
                        help='directory of testing model')
    args = parser.parse_args()
    return args


def compute_pred_prob(predicted):
    predicted = (predicted[0] + predicted[1] + predicted[2]) / 3
    return predicted.view(-1)


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
test_set = SignatureLoader(
    root='dataset/ChiSig/ChiSig_resize/', train=False)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=2*BATCH_SIZE, shuffle=False)
args = parse_args()
assert args.model_dir != '', 'model_dir is required'

model = net().to(device)
model.load_state_dict(torch.load(args.model_dir))

predicted = []
labels = []
with torch.no_grad():
    accuracys = []
    for inputs_, labels_ in tqdm(test_loader):
        labels_ = labels_.float()
        inputs_, labels_ = inputs_.to(device), labels_.to(device)
        predicted_ = model(inputs_)
        predicted += list(compute_pred_prob(predicted_).detach().cpu().numpy())
        labels += list(labels_.detach().cpu().numpy())
        accuracys.append(compute_accuracy(predicted_, labels_))
    accuracy_ = sum(accuracys) / len(accuracys)
print(f'test accuracy:{accuracy_:%}')

fpr, tpr, thresholds = metrics.roc_curve(labels, predicted)
print(f'AUC: {metrics.auc(fpr, tpr)}')
plot_roc_curve(fpr, tpr, 'BHSig-H')

plot_far_frr_curve(fpr=fpr, fnr=1-tpr, threshold=thresholds)
