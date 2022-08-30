import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_loss = nn.BCELoss()

    def forward(self, x, y, z, label):
        alpha_1, alpha_2, alpha_3 = 0.3, 0.4, 0.3
        label = label.view(-1, 1)
        # print(max(x), max(label))
        loss_1 = self.bce_loss(x, label)
        loss_2 = self.bce_loss(y, label)
        loss_3 = self.bce_loss(z, label)
        return torch.mean(alpha_1*loss_1 + alpha_2*loss_2 + alpha_3*loss_3)
