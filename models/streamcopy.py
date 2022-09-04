import torch
import torch.nn as nn

import cv2


class stream(nn.Module):
    def __init__(self):
        def block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_f, out_f, 3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2, stride=2)
            )
        super(stream, self).__init__()

        self.blocks = [
            block(1, 32).to(device),
            block(2, 64).to(device),
            block(64, 96).to(device),
            block(96, 128).to(device)
        ]

        self.Conv_32 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.Conv_64 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.Conv_96 = nn.Conv2d(96, 96, 3, stride=1, padding=1)
        self.Conv_128 = nn.Conv2d(128, 128, 3, stride=1, padding=1)

        self.fc_32 = nn.Linear(32, 32)
        self.fc_64 = nn.Linear(64, 64)
        self.fc_96 = nn.Linear(96, 96)
        self.fc_128 = nn.Linear(128, 128)

        self.max_pool = nn.MaxPool2d(2, stride=2)

    def forward(self, reference, inverse):
        for i in range(4):
            reference = self.blocks[i][0](reference)
            reference = self.blocks[i][1](reference)
            inverse = self.blocks[i][0](inverse)
            inverse = self.blocks[i][1](inverse)
            inverse = self.blocks[i][2](inverse)
            inverse = self.blocks[i][3](inverse)
            inverse = self.blocks[i][4](inverse)
            reference = self.attention(inverse, reference)
            reference = self.blocks[i][2](reference)
            reference = self.blocks[i][3](reference)
            reference = self.blocks[i][4](reference)

        return reference, inverse

    def attention(self, inverse, discrimnative):
        GAP = nn.AdaptiveAvgPool2d((1, 1))
        sigmoid = nn.Sigmoid()

        up_sample = nn.functional.interpolate(
            inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')

        conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
        g = conv(up_sample)
        g = sigmoid(g)

        tmp = g * discrimnative + discrimnative
        f = GAP(tmp)
        f = f.view(f.size()[0], 1, f.size()[1])

        fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
        f = fc(f)
        f = sigmoid(f)
        f = f.view(-1, f.size()[2], 1, 1)

        out = tmp * f

        return out


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    model = stream()
    # r, i = model(torch.ones(1,3,32,32), torch.ones(1,3,32,32))
    # print(r.size(), i.size())
    img = cv2.imread('dataset/BHSig260/Bengali_56x250/001/B-S-001-G-01.tif')
    print(model.blocks[0][0](torch.FloatTensor(img).to(device)))
