import os
import torch
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def plot_roc_curve(fpr, tpr, filename):
    plt.plot(fpr, tpr, color='red', label='ROC')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve with AUC: {auc:.2%}')
    plt.legend()
    plt.savefig(os.path.join('imgs', 'plot', filename))
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
    plt.savefig(os.path.join('imgs', 'plot', 'curve.jpg'))


def draw_failed_sample(samples):
    rows = 1
    columns = 2
    
    folder_name = os.path.join('imgs', 'failed_sample', datetime.now().strftime('%m%d_%H%M%S'))
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
