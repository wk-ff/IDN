import cv2
import os
import numpy as np
import random


# normalize each image by dividing the pixel values with the 
# standard deviation of the pixel values of the images in a dataset

# def resize_img(path):
# 	# threshold = 220
# 	try:
# 		img = cv2.imread('CEDAR/full_forg/' + path, 0)
# 		dst = cv2.resize(img, (220, 155), cv2.INTER_LINEAR)
# 		cv2.imwrite('CEDAR/full_forg_gray_115x220/{}'.format(path), dst)
# 	except:
# 		print(path)

# path = 'CEDAR/full_forg'
# for p in os.listdir(path):
# 	resize_img(p)


# img = cv2.imread('dataset/original_2_9.png', 0)
# # img_ = cv2.resize(img, (img.shape[0]*2, img.shape[1]*2), cv2.INTER_NEAREST)
# print(img.shape)
# cv2.imshow('origin', img)
# cv2.imshow('resize', 255 - img)

# cv2.waitKey()


with open('CEDAR/gray_train.txt', 'w') as f:
	for i in range(1, 51):
		for j in range(1, 25):
			for k in range(j+1, 25):
				f.write('full_org_gray_115x220/original_{0}_{1}.png full_org_gray_115x220/original_{0}_{2}.png 1\n'.format(i, j, k))
		org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
		for (j, k) in random.choices(org_forg, k=276):
			f.write('full_org_gray_115x220/original_{0}_{1}.png full_forg_gray_115x220/forgeries_{0}_{2}.png 0\n'.format(i, j, k))

with open('CEDAR/gray_test.txt', 'w') as f:
	for i in range(51, 56):
		for j in range(1, 25):
			for k in range(j+1, 25):
				f.write('full_org_gray_115x220/original_{0}_{1}.png full_org_gray_115x220/original_{0}_{2}.png 1\n'.format(i, j, k))
		org_forg = [(j,k) for j in range(1, 25) for k in range(1, 25)]
		for (j, k) in random.choices(org_forg, k=276):
			f.write('full_org_gray_115x220/original_{0}_{1}.png full_forg_gray_115x220/forgeries_{0}_{2}.png 0\n'.format(i, j, k))

