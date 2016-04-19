from __future__ import print_function
from __future__ import division

import os
import glob
import pickle
import random
import numpy as np
import pandas as pd

from tqdm import tqdm

from sklearn.preprocessing import OneHotEncoder
from skimage.io import imread, imsave
from scipy.misc import imresize

SUBSET = False
DOWNSAMPLE = 20
NUM_CLASSES = 10

WIDTH, HEIGHT = 640 // DOWNSAMPLE, 480 // DOWNSAMPLE

def load_image(path):
    img = imread(path)
    img = imresize(img, (HEIGHT, WIDTH))
    return img

def load_train(base):
    driver_imgs_list = pd.read_csv('driver_imgs_list.csv')
    driver_imgs_grouped = driver_imgs_list.groupby('classname')

    X_train = []
    y_train = []
    driver_ids = []

    print('Reading train images...')
    for j in range(NUM_CLASSES):
        print('Loading folder c{}...'.format(j))
        driver_ids_group = driver_imgs_grouped.get_group('c{}'.format(j))
        paths = os.path.join(base, 'c{}/'.format(j)) + driver_ids_group.img

        if SUBSET:
            paths = paths[:100]
            driver_ids_group = driver_ids_group.iloc[:100]

        driver_ids += driver_ids_group['subject'].tolist()

        for i, path in tqdm(enumerate(paths), total=len(paths)):
            img = load_image(path)
            if i == 0:
                imsave('c{}.jpg'.format(j), img)
            img = img.swapaxes(2, 0)

            X_train.append(img)
            y_train.append(j)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    y_train = OneHotEncoder(n_values=NUM_CLASSES) \
        .fit_transform(y_train.reshape(-1, 1)) \
        .toarray()

    return X_train, y_train, driver_ids

def load_test(base):
    X_test = []
    X_test_id = []
    paths = glob.glob(os.path.join(base, '*.jpg'))

    if SUBSET:
        paths = paths[:100]

    print('Reading test images...')
    for i, path in tqdm(enumerate(paths), total=len(paths)):
        img_id = os.path.basename(path)
        img = load_image(path)
        img = img.swapaxes(2, 0)

        X_test.append(img)
        X_test_id.append(img_id)

    X_test = np.array(X_test)
    X_test_id = np.array(X_test_id)

    return X_test, X_test_id

X_train, y_train, driver_ids = load_train('imgs/train/')
X_test, X_test_ids = load_test('imgs/test/')

if SUBSET:
    dest = 'data_{}_subset.pkl'.format(DOWNSAMPLE)
else:
    dest = 'data_{}.pkl'.format(DOWNSAMPLE)

with open(dest, 'wb') as f:
    pickle.dump((X_train, y_train, X_test, X_test_ids, driver_ids), f)
