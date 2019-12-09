import torch
import torch.nn as nn

import cv2

class stream(nn.Module):
	def __init__(self):
		super(stream, self).__init__()

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
			print(reference.dtype)
			reference = self.stream[0 + i * 5](reference)
			reference = self.stream[1 + i * 5](reference)
			inverse = self.stream[0 + i * 5](inverse)
			inverse = self.stream[1 + i * 5](inverse)
			inverse = self.stream[2 + i * 5](inverse)
			inverse = self.stream[3 + i * 5](inverse)
			inverse = self.stream[4 + i * 5](inverse)
			reference = self.attention(inverse, reference)
			reference = self.stream[2 + i * 5](reference)
			reference = self.stream[3 + i * 5](reference)
			reference = self.stream[4 + i * 5](reference)
			

		return reference, inverse


	def attention(self, inverse, discrimnative):
		Conv = nn.Sequential(
			nn.Conv2d(inverse.size()[1], inverse.size()[1], 3, stride=1, padding=1),
			nn.Sigmoid()
		)
		GAP = nn.AdaptiveAvgPool2d((1, 1))
		fc = nn.Sequential(
			nn.Linear(inverse.size()[1], inverse.size()[1]),
			nn.Sigmoid()
		)

		# print(inverse.size(), discrimnative.size())
		# up_sample = cv2.resize(inverse, (inverse.size()[2]*2, inverse.size()[3]*2), interpolation=cv2.INTER_NEAREST)
		up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
		g = Conv(up_sample)
		# print(g.size(), discrimnative.size())
		tmp = g * discrimnative + discrimnative
		f = GAP(tmp)
		f = f.view(f.size()[0], 1, f.size()[1])
		
		f = fc(f)
		f = f.view(-1, f.size()[2], 1, 1)
		# print(tmp.size(), f.size())
		out = tmp * f

		return out


if __name__ == '__main__':
	model = stream()
	# r, i = model(torch.ones(1,3,32,32), torch.ones(1,3,32,32))
	# print(r.size(), i.size())
	x = {}
