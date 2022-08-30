from PIL import Image
import os
import numpy as np
import random


def resize_img(root, w, h):
    os.mkdir(f'{root}_{h}x{w}')
    for filename in os.listdir(root):
        with Image.open(f'{root}/{filename}') as img:
            img = img.resize((w, h))
            img.save(f'{root}_{h}x{w}/{filename}')


def resize_BHSig_img(root, w, h, size):
    os.mkdir(f'{root}_{h}x{w}')

    for i in range(1, size+1):
        os.mkdir(f'{root}_{h}x{w}/{i:03}')
        for filename in os.listdir(f'{root}/{i:03}'):
            with Image.open(f'{root}/{i:03}/{filename}') as img:
                img = img.resize((w, h))
                img.save(
                    f'{root}_{h}x{w}/{i:03}/{filename[:4]}{i:03}{filename[-9:]}')


def generate_CEDAR_pairs():
    # 55 individuals, each has 24 genuine and 24 forged
    # train : test = 50 : 5

    def generate(file, i):
        # reference-genuine pairs
        for j in range(1, 25):
            for k in range(j+1, 25):
                file.write(
                    f'full_org_115x220/original_{i}_{j}.png full_org_115x220/original_{i}_{k}.png 1\n')
        # reference-forgered pairs
        org_forg = [(j, k) for j in range(1, 25) for k in range(1, 25)]
        for (j, k) in random.choices(org_forg, k=276):
            file.write(
                f'full_org_115x220/original_{i}_{j}.png full_forg_115x220/forgeries_{i}_{k}.png 0\n')

    with open('CEDAR/train_pairs.txt', 'w') as f:
        for i in range(1, 51):
            generate(f, i)

    with open('CEDAR/test_pairs.txt', 'w') as f:
        for i in range(51, 56):
            generate(f, i)


def generate_BHSigB_pairs(root):
    # 100 individuals, each has 24 genuine and 30 forged
    # train : test = 50 : 50

    def generate(file, i):
        # reference-genuine pairs
        for j in range(1, 25):
            for k in range(j+1, 25):
                file.write(
                    f'{i:03}/B-S-{i:03}-G-{j:02}.tif {i:03}/B-S-{i:03}-G-{k:02}.tif 1\n')
        # reference-forgered pairs
        org_forg = [(j, k) for j in range(1, 25) for k in range(1, 31)]
        for (j, k) in random.choices(org_forg, k=276):
            file.write(
                f'{i:03}/B-S-{i:03}-G-{j:02}.tif {i:03}/B-S-{i:03}-F-{k:02}.tif 0\n')

    with open(f'{root}/train_pairs.txt', 'w') as f:
        for i in range(1, 51):
            generate(f, i)

    with open(f'{root}/test_pairs.txt', 'w') as f:
        for i in range(51, 101):
            generate(f, i)


def generate_BHSigH_pairs(root):
    # 160 individuals, each has 24 genuine and 30 forged
    # train : test = 100 : 60

    def generate(file, i):
        # reference-genuine pairs
        for j in range(1, 25):
            for k in range(j+1, 25):
                file.write(
                    f'{i:03}/H-S-{i:03}-G-{j:02}.tif {i:03}/H-S-{i:03}-G-{k:02}.tif 1\n')
        # reference-forgered pairs
        org_forg = [(j, k) for j in range(1, 25) for k in range(1, 31)]
        for (j, k) in random.choices(org_forg, k=276):
            file.write(
                f'{i:03}/H-S-{i:03}-G-{j:02}.tif {i:03}/H-S-{i:03}-F-{k:02}.tif 0\n')

    with open(f'{root}/train_pairs.txt', 'w') as f:
        for i in range(1, 101):
            generate(f, i)

    with open(f'{root}/test_pairs.txt', 'w') as f:
        for i in range(101, 161):
            generate(f, i)


if __name__ == '__main__':
    # resize_img('CEDAR/full_org', 220, 115)
    # resize_img('CEDAR/full_forg', 220, 115)
    # generate_CEDAR_pairs()
    resize_BHSig_img('BHSig260/Bengali', 250, 56, 100)
    resize_BHSig_img('BHSig260/Hindi', 250, 56, 160)
    generate_BHSigH_pairs('BHSig260/Hindi_56x250')
    generate_BHSigB_pairs('BHSig260/Bengali_56x250')
