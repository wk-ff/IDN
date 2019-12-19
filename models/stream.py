import torch
import torch.nn as nn

import cv2

class stream(nn.Module):
	def __init__(self):
		super(stream, self).__init__()

		self.stream = nn.Sequential(
			nn.Conv2d(1, 32, 3, stride=1, padding=1),
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
		# Conv = nn.Sequential(
		# 	nn.Conv2d(inverse.size()[1], inverse.size()[1], 3, stride=1, padding=1),
		# 	nn.Sigmoid()
		# )
		GAP = nn.AdaptiveAvgPool2d((1, 1))
		sigmoid = nn.Sigmoid()
		# fc = nn.Sequential(
		# 	nn.Linear(inverse.size()[1], inverse.size()[1]),
		# 	nn.Sigmoid()
		# )

		# print(inverse.size(), discrimnative.size())
		up_sample = nn.functional.interpolate(inverse, (discrimnative.size()[2], discrimnative.size()[3]), mode='nearest')
		# g = self.Conv(up_sample)
		conv = getattr(self, 'Conv_' + str(up_sample.size()[1]), 'None')
		g = conv(up_sample)
		g = sigmoid(g)
		# print(g.size(), discrimnative.size())
		tmp = g * discrimnative + discrimnative
		f = GAP(tmp)
		f = f.view(f.size()[0], 1, f.size()[1])
		
		# f = self.fc(f)
		fc = getattr(self, 'fc_' + str(f.size(2)), 'None')
		f = fc(f)
		f = sigmoid(f)
		f = f.view(-1, f.size()[2], 1, 1)
		# print(tmp.size(), f.size())
		out = tmp * f

		return out


if __name__ == '__main__':
	model = stream()
	# r, i = model(torch.ones(1,3,32,32), torch.ones(1,3,32,32))
	# print(r.size(), i.size())
	x = {}
