from torch.utils import data
import torch
import cv2
import numpy as np


class SignatureDataset(data.Dataset):
    def __init__(self, root='dataset/CEDAR/', train=True):
        super().__init__()
        if train:
            path = root + 'train_pairs.txt'
        else:
            path = root + 'test_pairs.txt'

        with open(path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        for line in lines:
            refer, test, label = line.split()
            try:
                refer_img = cv2.imread(root + refer, 0)
                test_img = cv2.imread(root + test, 0)
                refer_img = refer_img.reshape(-1,
                                              refer_img.shape[0], refer_img.shape[1])
                test_img = test_img.reshape(-1,
                                            test_img.shape[0], test_img.shape[1])
            except:
                print(root+refer)
                raise NotImplementedError

            refer_test = np.concatenate((refer_img, test_img), axis=0)
            self.datas.append(refer_test)
            self.labels.append(int(label))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return torch.FloatTensor(self.datas[index]), float(self.labels[index])
