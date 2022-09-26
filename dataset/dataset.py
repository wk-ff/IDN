from torch.utils import data
import torch
import cv2
import numpy as np


class SignatureLoader(data.Dataset):
    def __init__(self, root='dataset/CEDAR/', train=True):
        super().__init__()
        self.root = root

        if train:
            path = root + 'train_pairs.txt'
        else:
            path = root + 'test_pairs.txt'

        with open(path, 'r') as f:
            lines = f.readlines()

        self.refers = []
        self.tests = []
        self.labels = []
        for line in lines:
            refer, test, label = line.split()

            self.refers.append(refer)
            self.tests.append(test)
            self.labels.append(float(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        refer_img = cv2.imread(self.root + self.refers[index], 0)
        test_img = cv2.imread(self.root + self.tests[index], 0)
        refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])
        test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])
        refer_test = np.concatenate((refer_img, test_img), axis=0)
        return torch.FloatTensor(refer_test), self.labels[index]
