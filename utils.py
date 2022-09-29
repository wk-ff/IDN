import argparse
import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from models.net import net
from dataset.dataset import SignatureLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
post_fix = datetime.now().strftime('%m%d_%H%M%S')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--attn', default='IDN', type=str, help='type of attention module')
    parser.add_argument('--model_dir', default='', type=str,
                        help='directory of testing model')
    parser.add_argument('--n_epoch', default=1, type=int,
                        help='number of epoch to train')
    parser.add_argument('--dataset_dir', default='dataset/CEDAR/', type=str,
                        help='directory of dataset')
    parser.add_argument('--model_prefix', default='CEDAR', type=str,
                        help='prefix of model name')
    args = parser.parse_args()
    return args


def plot_roc_curve(auc, fpr, tpr, filename):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with AUC: {auc:.2%}')
    plt.legend()
    plt.savefig(os.path.join('imgs', 'plot', f'{filename}_{post_fix}'))
    plt.show()


def plot_far_frr_curve(fpr, fnr, threshold, filename):
    _, ax = plt.subplots()

    ax.plot(threshold, fpr, 'r--', label='FAR')
    ax.plot(threshold, fnr, 'g--', label='FRR')
    plt.xlabel('Threshold')
    plt.xticks([])
    plt.ylabel('far/frr')

    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    plt.plot(threshold[eer_idx], fpr[eer_idx], 'bo', label='EER')

    ax.legend(loc='upper right', shadow=True, fontsize='x-large')

    plt.title(f'{filename} EER: {fpr[eer_idx]:.2%}')
    plt.savefig(os.path.join('imgs', 'plot', f'curve_{filename}_{post_fix}'))


def draw_failed_sample(samples):
    rows = 1
    columns = 2

    folder_name = os.path.join(
        'imgs', 'failed_sample', datetime.now().strftime('%m%d_%H%M%S'))
    os.mkdir(folder_name)
    os.mkdir(os.path.join(folder_name, 'fp'))
    os.mkdir(os.path.join(folder_name, 'fn'))

    for i, (input, label) in enumerate(samples):
        reference = input[0]
        test = input[1]
        label = int(label)

        fig = plt.figure(figsize=(8, 3))
        plt.suptitle(f'Ground Truth: {label} Predict: {int(not label)}')

        fig.add_subplot(rows, columns, 1)
        plt.imshow(reference, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Reference')

        fig.add_subplot(rows, columns, 2)
        plt.imshow(test, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.title('Test')

        plt.savefig(os.path.join(folder_name, 'fn' if label ==
                    1 else 'fp', f'index_{i}.jpg'))
        plt.close()


def visualize_stream(model_dir, dataset_dir, filename, data_idx):
    # get model
    model = net().to(device)
    model.load_state_dict(torch.load(model_dir))

    # use forward hooks to get activation map
    # https://discuss.pytorch.org/t/visualize-feature-map/29597/2
    activation = {}

    def get_activation(name):
        def hook(model, input, output):
            activation[name] = output.detach().cpu().numpy()
        return hook

    for i in range(4):
        model.stream.stream[4 + i * 5] \
                    .register_forward_hook(get_activation(f'block{i + 1}'))

    # get data
    dataset = SignatureLoader(root=dataset_dir, train=False)
    data, _ = dataset[data_idx]
    data = data.to(device).unsqueeze_(0)
    _ = model(data)

    # plot 4 blocks' feature map
    fig, ax = plt.subplots(4)
    for i in range(4):
        act = activation[f'block{i + 1}'].squeeze()
        ax[i].imshow(np.mean(act, axis=0))
        ax[i].set_title(f'block{i + 1}')
        ax[i].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join('imgs', 'vis', filename), dpi=600)
