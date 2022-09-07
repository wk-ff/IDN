from PIL import Image
import os
import pandas as pd


def resize_img(root, w=220, h=115):
    os.mkdir(f'{root}_resize')
    for filename in os.listdir(root):
        with Image.open(f'{root}/{filename}') as img:
            img = img.resize((w, h))
            img.save(f'{root}_resize/{filename}')


def resize_BHSig_img(root, size, w=250, h=56):
    os.mkdir(f'{root}_resize')

    for i in range(1, size+1):
        os.mkdir(f'{root}_resize/{i:03}')
        for filename in os.listdir(f'{root}/{i:03}'):
            with Image.open(f'{root}/{i:03}/{filename}') as img:
                img = img.resize((w, h))
                img.save(
                    f'{root}_resize/{i:03}/{filename[:4]}{i:03}{filename[-9:]}')


def resize_SigComp_test_img(root, w=220, h=115):
    os.mkdir(f'{root}_resize')

    for i in range(11, 21):
        os.mkdir(f'{root}_resize/{i:03}')
        for filename in os.listdir(f'{root}/{i:03}'):
            with Image.open(f'{root}/{i:03}/{filename}') as img:
                img = img.resize((w, h))
                img.save(f'{root}_resize/{i:03}/{filename}')


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


def generate_SigComp_pairs(root: str):
    f = open(os.path.join(root, 'train_pairs.txt'), 'w+')
    path_train_g = os.path.join(root, 'train/Offline_Genuine_resize/')
    path_train_f = os.path.join(root, 'train/Offline_Forgeries_resize/')

    # list forgeries fId pair
    genuine_names = os.listdir(path_train_g)
    gId_pair = {}
    for (k, v) in [f.split('_') for f in genuine_names]:
        gId_pair.setdefault(k, []).append(v)
    for k in sorted(gId_pair.keys()):
        gId_pair[k] = sorted(gId_pair[k], key = lambda v: int(v.split('.')[0]))

    forgeries_names = os.listdir(path_train_f)
    fId_pair = {}
    for (k, v) in set([(f[4:7], f[:4]) for f in forgeries_names]):
        fId_pair.setdefault(k, []).append(v)

    # generate train pair
    for i in range(1, 11):
        iStr = f'{i:03}'
        # reference-genuine pairs
        for gIdNoIdx in range(len(gId_pair[iStr]) - 1):
            for j in range(gIdNoIdx + 1, len(gId_pair[iStr])):
                f.write(f'{path_train_g}{iStr}_{gId_pair[iStr][gIdNoIdx]} {path_train_g}{iStr}_{gId_pair[iStr][j]} 1\n')
        # reference-forged pairs
        for gIdNo in gId_pair[iStr]:
            for fid in fId_pair[iStr]:
                for forged_name in list(filter(lambda f: f.startswith(f'{fid}{iStr}'), forgeries_names)):
                    f.write(f'{path_train_g}{iStr}_{gIdNo} {path_train_f}{forged_name} 0\n')

    f = open(os.path.join(root, 'test_pairs.txt'), 'w+')
    path_test_r = os.path.join(root, 'test/Ref(115)_resize/')
    path_test_q = os.path.join(root, 'test/Questioned(487)_resize/')

    # generate test pair
    for i in range(11, 21):
        for ref_name in os.listdir(os.path.join(path_test_r, f'{i:03}')):
            for test_name in os.listdir(os.path.join(path_test_q, f'{i:03}')):
                flag = 1 if len(test_name) < 11 else 0
                f.write(f'{path_test_r}{i:03}/{ref_name} {path_test_q}{i:03}/{test_name} {flag}\n')

if __name__ == '__main__':
    # resize_img('CEDAR/full_org')
    # resize_img('CEDAR/full_forg')
    # resize_BHSig_img('BHSig260/Bengali', 100)
    # resize_BHSig_img('BHSig260/Hindi', 160)
    # generate_pairs('CEDAR', 'C', 51)
    # generate_pairs('BHSig260/Bengali_resize', 'B', 51)
    # generate_pairs('BHSig260/Hindi_resize', 'H', 101)
    # resize_img('ChiSig')
    # generate_Chisig_pairs('ChiSig/ChiSig_resize', 400)
    # resize_img('SigComp2011/train/Offline_Genuine')
    # resize_img('SigComp2011/train/Offline_Forgeries')
    # resize_SigComp_test_img('SigComp2011/test/Ref(115)')
    # resize_SigComp_test_img('SigComp2011/test/Questioned(487)')
    # generate_SigComp_pairs('SigComp2011')
