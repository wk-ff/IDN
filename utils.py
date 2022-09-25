import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
post_fix = datetime.now().strftime('%m%d_%H%M%S')

def plot_roc_curve(auc, fpr, tpr, filename):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with AUC: {auc:.2%}')
    plt.legend()
    plt.savefig(os.path.join('imgs', 'plot', f'{filename}_{post_fix}'))
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
    plt.savefig(os.path.join('imgs', 'plot', f'curve_{post_fix}'))


def draw_failed_sample(samples):
    rows = 1
    columns = 2
    
    folder_name = os.path.join('imgs', 'failed_sample', post_fix)
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

        plt.savefig(os.path.join(folder_name, 'fn' if label == 1 else 'fp', f'index_{i}.jpg'))
        plt.close()