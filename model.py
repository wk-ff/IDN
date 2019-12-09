import torch
import torch.nn as nn

import cv2

class model(nn.Module):
	def __init__(self):
		super(model, self).__init__()

		self.stream = nn.Sequential(
			nn.Conv2d(3, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(32, 32, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(32, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 64, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(64, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(96, 96, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2),

			nn.Conv2d(96, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 128, 3, stride=1, padding=1),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(2, stride=2)
			)
		
		self.max_pool = nn.MaxPool2d(2, stride=2)

	def forward(self, reference, inverse):
		for i in range(4):
			reference = self.stream[0 + i](reference)
			reference = self.stream[1 + i](reference)
			inverse = self.stream[0 + i](inverse)
			inverse = self.stream[1 + i](inverse)
			inverse = self.stream[2 + i](inverse)
			inverse = self.stream[3 + i](inverse)
			reference = self.attention(inverse, reference)
			reference = self.stream[2 + i](reference)
			reference = self.stream[3 + i](reference)
			reference = self.stream[4 + i](reference)
			inverse = self.stream[4 + i](inverse)

		return reference, inverse

		

	def attention(self, inverse, discrimnative):
		Conv = nn.Sequential(
			nn.Conv2d(inverse.size()[1], inverse.size()[1], 3, stride=1, padding=1),
			nn.Sigmoid()
		)
		GAP = nn.AdaptiveAvgPool2d((1, 1))
		fc = nn.Linear(inverse.size()[1], inverse.size()[1])

		print(inverse.size())
		up_sample = cv2.resize(inverse, (inverse.size()[2]*2, inverse.size()[3]*2), interpolation=cv2.INTER_NEAREST)
		g = Conv(up_sample)
		tmp = g * discrimnative + discrimnative
		f = GAP(tmp)
		f = fc(f)
		out = tmp * f

		return out


if __name__ == '__main__':
	# model = model()
	# r, i = model(torch.ones(1,3,32,32), torch.ones(1,3,32,32))
	x = torch.arange(48).view(2,4,6)
	print(x)
	print(x.resize_(2,2,3))
