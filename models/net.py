import torch
import torch.nn as nn
from models.stream import stream

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()

        self.stream = stream()
        self.GAP = nn.AdaptiveAvgPool2d((1,1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        

    def forward(self, inputs):
        reference = inputs[:, 0:3, :, :]
        reference_inverse = 255 - reference
        test = inputs[:, 3:6, :, :]
        test_inverse = 255 - test

        reference, reference_inverse = self.stream(reference, reference_inverse)
        test, test_inverse = self.stream(test, test_inverse)

        cat_1 = torch.cat((test, reference_inverse), dim=1)
        cat_2 = torch.cat((reference, test), dim=1)
        cat_3 = torch.cat((reference, test_inverse), dim=1)

        out_1 = self.sub_forward(cat_1)
        out_2 = self.sub_forward(cat_2)
        out_3 = self.sub_forward(cat_3)

        return out_1, out_2, out_3
    
    def sub_forward(self, inputs):
        out = self.GAP(inputs)
        out = out.view(-1, 1, inputs.size()[1])
        out = self.classifier(out)

        return out

if __name__ == '__main__':
    net = net()
    x = torch.ones(1, 3, 32, 32)
    y = torch.ones(1, 3, 32, 32)
    x_ = torch.ones(1, 3, 32, 32)
    y_ = torch.ones(1, 3, 32, 32)
    out_1, out_2, out_3 = net(x, y, x_, y_)