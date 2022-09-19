import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import models

import torchvision.transforms as transforms
import torchvision.datasets as dataset

import matplotlib.pyplot as plt
import numpy as np
import cv2

from models.net import net
from models.stream import stream


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_roc_curve(auc, fper, tper, filename):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with AUC: {auc:.2%}')
    plt.legend()
    plt.savefig(filename)
    plt.show()