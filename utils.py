import torch

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import LineString

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


def plot_far_frr_curve(far, frr, threshold):
    _, ax = plt.subplots()

    ax.plot(threshold, far, 'r--', label='FAR')
    ax.plot(threshold, frr, 'g--', label='FRR')
    plt.xlabel('Threshold')

    first_line = LineString(np.column_stack((threshold, far)))
    second_line = LineString(np.column_stack((threshold, frr)))
    eer = first_line.intersection(second_line)
    plt.plot(15, *eer.xy, 'bo', label='EER')

    ax.legend(loc='upper center', shadow=True, fontsize='x-large')

    plt.show()
