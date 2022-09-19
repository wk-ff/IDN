import torch

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import LineString

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_roc_curve(fpr, tpr, filename):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic Curve')
    plt.legend()
    plt.savefig(filename)
    plt.show()


def plot_far_frr_curve(fpr, fnr, threshold):
    _, ax = plt.subplots()

    ax.plot(threshold, fpr, 'r--', label='FAR')
    ax.plot(threshold, fnr, 'g--', label='FRR')
    plt.xlabel('Threshold')
    plt.xticks([])
    plt.ylabel('far/frr')

    eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
    plt.plot(threshold[eer_idx], fpr[eer_idx], 'bo', label='EER')

    ax.legend(loc='upper right', shadow=True, fontsize='x-large')

    plt.title(f'EER: {fpr[eer_idx]:.2%}')
    plt.savefig('curve.jpg')
