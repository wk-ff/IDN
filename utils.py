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


def plot_roc_curve(fper, tper, filename):
    plt.plot(fper, tper, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def visualize_single_stream(model_dir, refer_dir, test_dir):

    # get model
    model = net().to(device)
    model.load_state_dict(torch.load(model_dir))

    # get image
    refer_img = cv2.imread(refer_dir, 0)
    test_img = cv2.imread(test_dir, 0)
    refer_img = refer_img.reshape(-1, refer_img.shape[0], refer_img.shape[1])
    test_img = test_img.reshape(-1, test_img.shape[0], test_img.shape[1])
    refer_test = torch.FloatTensor(np.concatenate(
        (refer_img, test_img), axis=0)).to(device)

    # get Conv2d layers
    no_of_layers = 0
    conv_layers = []
    for child in list(model.stream.stream.children()):
        if type(child) == nn.Conv2d:
            no_of_layers += 1
            conv_layers.append(child)
        elif type(child) == nn.Sequential:
            for layer in child.children():
                if type(layer) == nn.Conv2d:
                    no_of_layers += 1
                    conv_layers.append(layer)
    print(no_of_layers)

    # put image into stream con
    results = [conv_layers[0](refer_test)]
    for i in range(1, len(conv_layers)):
        results.append(conv_layers[i](results[-1]))
    outputs = results

    # visualize
    for num_layer in range(len(outputs)):
        plt.figure(figsize=(50, 10))
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data
        print("Layer ", num_layer + 1)
        for i, filter in enumerate(layer_viz):
            if i == 16:
                break
            plt.subplot(2, 8, i + 1)
            plt.imshow(filter)
            plt.axis("off")
        plt.show()
        plt.close()


# if __name__ == '__main__':
    # model_dir = 'BHSigH_model_90.103161%.pth'
    # image_root = 'dataset/BHSig260/Bengali_56x250/001/'
    # refer_dir = f'{image_root}B-S-001-G-01.tif'
    # test_dir = f'{image_root}B-S-001-F-01.tif'
    # visualize_single_stream(model_dir, refer_dir, test_dir)
