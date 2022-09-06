from PIL import Image
import os
import pandas as pd


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


def generate_Chisig_pairs(root: str, cutting_point: int):

    def generate(mode: str, names):
        f = open(f'{root}/{mode}_pairs.txt', 'w+')
        for name in names:
            group = df.get_group(name)
            subgroup = group.groupby('id')
            ids = list(group.id.unique())
            for refIdx in range(len(ids)):
                # must be forged from id-100
                if ids[refIdx]> 100:
                    break
                refNos = list(subgroup.get_group(ids[refIdx])['no'])
                # reference-genuine pairs
                for i in range(len(refNos)):
                    for j in range(i + 1, len(refNos)):
                        f.write(f'{name}-{ids[refIdx]}-{refNos[i]}.jpg {name}-{ids[refIdx]}-{refNos[j]}.jpg 1\n')
                # reference-forged pairs
                for i in range(len(refNos)):
                    for testIdx in range(refIdx + 1, len(ids)):
                        testNos = list(subgroup.get_group(ids[testIdx])['no'])
                        for j in range(len(testNos)):
                            f.write(f'{name}-{ids[refIdx]}-{refNos[i]}.jpg {name}-{ids[testIdx]}-{testNos[j]}.jpg 0\n')
    
    data = list(filter(lambda dir: dir.endswith('.jpg'), os.listdir(root)))
    data_tuple = [(d[0], int(d[1]), d[2].split('.')[0]) for d in [d.split('-') for d in data]]
    df = pd.DataFrame(data_tuple, columns=['name', 'id', 'no'])
    df = df.sort_values(by=['name', 'id', 'no']).groupby('name')

    generate('train', list(df.groups.keys())[:cutting_point])
    generate('test', list(df.groups.keys())[cutting_point + 1:])


if __name__ == '__main__':
    # resize_img('CEDAR/full_org', 220, 115)
    # resize_img('CEDAR/full_forg', 220, 115)
    # resize_BHSig_img('BHSig260/Bengali', 250, 56, 100)
    # resize_BHSig_img('BHSig260/Hindi', 250, 56, 160)
    # generate_pairs('CEDAR', 'C', 51)
    # generate_pairs('BHSig260/Bengali_resize', 'B', 51)
    # generate_pairs('BHSig260/Hindi_resize', 'H', 101)
    # resize_img('ChiSig', 220, 115)
    generate_Chisig_pairs('ChiSig/ChiSig_resize', 400)
