from PIL import Image
import os
import numpy as np
import random


def resize_img(root):

	for filename in os.listdir(root):
		with Image.open(f'{root}/{filename}') as img:
			img = img.resize((220, 155))
			img.save(f'{root}_115x220/{filename}')


def generate_CEDAR_pairs():

	with open('CEDAR/train_pairs.txt', 'w') as f:
		for i in range(1, 51):
			# reference-genuine pairs
			for j in range(1, 25):
				for k in range(j+1, 25):
					f.write(f'full_org_115x220/original_{i}_{j}.png full_org_115x220/original_{i}_{k}.png 1\n')
			# reference-forgered pairs
			org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
			for (j, k) in random.choices(org_forg, k=276):
				f.write(f'full_org_115x220/original_{i}_{j}.png full_forg_115x220/forgeries_{i}_{k}.png 0\n')

	with open('CEDAR/test_pairs.txt', 'w') as f:
		for i in range(51, 56):
			# reference-genuine pairs
			for j in range(1, 25):
				for k in range(j+1, 25):
					f.write(f'full_org_115x220/original_{i}_{j}.png full_org_115x220/original_{i}_{k}.png 1\n')
			# reference-forgered pairs
			org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
			for (j, k) in random.choices(org_forg, k=276):
				f.write(f'full_org_115x220/original_{i}_{j}.png full_forg_115x220/forgeries_{i}_{k}.png 0\n')


if __name__ == '__main__':
	resize_img('CEDAR/full_org')
	resize_img('CEDAR/full_forg')
	generate_CEDAR_pairs()