from torch.utils import data
import cv2
import numpy as np

class dataset(data.Dataset):
    def __init__(self, root='CEDAR/', train=True):
        super(dataset, self).__init__()
        if train:
            path = root + 'train.txt'
        else:
            path = root + 'test.txt'
        
        with open(path, 'r') as f:
            lines = f.readlines()

        self.labels = []
        self.datas = []
        for line in lines:
            refer, test, label = line.split()
            # print(root + refer)
            refer_img = cv2.imread(root + refer)
            test_img = cv2.imread(root + test)
            refer_test = np.concatenate((refer_img, test_img), axis=2)
            self.datas.append(refer_test.reshape(6, refer_test.shape[0], refer_test.shape[1]))
            self.labels.append(label)

        print(self.datas[0].shape)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.datas[index], self.labels[index]

# img = cv2.imread('dataset/original_2_9.png')
# print(img.shape)