from PIL import Image
import os
import numpy as np
import random


def resize_img(root, w, h):
    os.mkdir(f'{root}_resize')
    for filename in os.listdir(root):
        with Image.open(f'{root}/{filename}') as img:
            img = img.resize((w, h))
            img.save(f'{root}_resize/{filename}')


def resize_BHSig_img(root, w, h, size):
    os.mkdir(f'{root}_resize')

    for i in range(1, size+1):
        os.mkdir(f'{root}_resize/{i:03}')
        for filename in os.listdir(f'{root}/{i:03}'):
            with Image.open(f'{root}/{i:03}/{filename}') as img:
                img = img.resize((w, h))
                img.save(
                    f'{root}_resize/{i:03}/{filename[:4]}{i:03}{filename[-9:]}')


def generate_pairs(root: str, mode: str, cutting_point: int):
    '''
    Generate reference-test pair for dataset.

    Input:
        root: path of dataset
        mode: which dataset
        cutting_point: the cutting point to split dataset into train and test
    Output:
        None
    '''

    if mode == 'C':
        size = 55
    elif mode == 'B':
        size = 100
    elif mode == 'H':
        size = 160

    num_genuine = 24
    num_forged = 24 if mode == 'C' else 30

    def pair_string_genuine(i, j, k):
        if mode == 'C':
            return f'full_org_resize/original_{i}_{j}.png full_org_resize/original_{i}_{k}.png 1\n'
        else:
            return f'{i:03}/{mode}-S-{i:03}-G-{j:02}.tif {i:03}/{mode}-S-{i:03}-G-{k:02}.tif 1\n'

    def pair_string_forged(i, j, k):
        if mode == 'C':
            return f'full_org_resize/original_{i}_{j}.png full_forg_resize/forgeries_{i}_{k}.png 0\n'
        else:
            return f'{i:03}/{mode}-S-{i:03}-G-{j:02}.tif {i:03}/{mode}-S-{i:03}-F-{k:02}.tif 0\n'

    def generate(file, i):
        # reference-genuine pairs
        for j in range(1, num_genuine + 1):
            for k in range(j + 1, num_genuine + 1):
                file.write(pair_string_genuine(i, j, k))
        # reference-forgered pairs
        org_forg = [(j, k) for j in range(1, num_genuine + 1)
                    for k in range(1, num_forged + 1)]
        for (j, k) in org_forg:
            file.write(pair_string_forged(i, j, k))

    with open(f'{root}/train_pairs.txt', 'w') as f:
        for i in range(1, cutting_point):
            generate(f, i)

    with open(f'{root}/test_pairs.txt', 'w') as f:
        for i in range(cutting_point, size + 1):
            generate(f, i)


if __name__ == '__main__':
    resize_img('CEDAR/full_org', 220, 115)
    resize_img('CEDAR/full_forg', 220, 115)
    resize_BHSig_img('BHSig260/Bengali', 250, 56, 100)
    resize_BHSig_img('BHSig260/Hindi', 250, 56, 160)
    generate_pairs('CEDAR', 'C', 51)
    generate_pairs('BHSig260/Bengali_resize', 'B', 51)
    generate_pairs('BHSig260/Hindi_resize', 'H', 101)
