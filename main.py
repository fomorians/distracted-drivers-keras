from __future__ import print_function
from __future__ import division

import os
import time
import pickle
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.normalization import BatchNormalization

from sklearn.metrics import log_loss
from sklearn.cross_validation import LabelShuffleSplit

from utilities import write_submission, calc_geom, calc_geom_arr, mkdirp

TESTING = False

DATASET_PATH = os.environ.get('DATASET_PATH', 'dataset/data_20.pkl' if not TESTING else 'dataset/data_20_subset.pkl')

CHECKPOINT_PATH = os.environ.get('CHECKPOINT_PATH', 'checkpoints/')
SUMMARY_PATH = os.environ.get('SUMMARY_PATH', 'summaries/')
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/')

mkdirp(CHECKPOINT_PATH)
mkdirp(SUMMARY_PATH)
mkdirp(MODEL_PATH)

NB_EPOCHS = 10 if not TESTING else 1
MAX_FOLDS = 3
DOWNSAMPLE = 20

WIDTH, HEIGHT, NB_CHANNELS = 640 // DOWNSAMPLE, 480 // DOWNSAMPLE, 3
BATCH_SIZE = 50

with open(DATASET_PATH, 'rb') as f:
    X_train_raw, y_train_raw, X_test, X_test_ids, driver_ids = pickle.load(f)
_, driver_indices = np.unique(np.array(driver_ids), return_inverse=True)

predictions_total = [] # accumulated predictions from each fold
scores_total = [] # accumulated scores from each fold
num_folds = 0

def vgg_bn():
    model = Sequential()
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal', input_shape=(NB_CHANNELS, WIDTH, HEIGHT)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(64, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, subsample=(2, 2), init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Convolution2D(128, 3, 3, border_mode='same', init='he_normal'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='sigmoid', init='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax', init='he_normal'))
    model.compile(Adam(lr=1e-3), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

for train_index, valid_index in LabelShuffleSplit(driver_indices, n_iter=MAX_FOLDS, test_size=0.2, random_state=67):
    print('Fold {}/{}'.format(num_folds + 1, MAX_FOLDS))

    # skip fold if a checkpoint exists for the next one
    # next_checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds + 1))
    # if os.path.exists(next_checkpoint_path):
    #     print('Checkpoint exists for next fold, skipping current fold.')
    #     continue

    X_train, y_train = X_train_raw[train_index,...], y_train_raw[train_index,...]
    X_valid, y_valid = X_train_raw[valid_index,...], y_train_raw[valid_index,...]

    model = vgg_bn()

    model_path = os.path.join(MODEL_PATH, 'model_{}.json'.format(num_folds))
    with open(model_path, 'w') as f:
        f.write(model.to_json())

    # restore existing checkpoint, if it exists
    checkpoint_path = os.path.join(CHECKPOINT_PATH, 'model_{}.h5'.format(num_folds))
    if os.path.exists(checkpoint_path):
        print('Restoring fold from checkpoint.')
        model.load_weights(checkpoint_path)

    summary_path = os.path.join(SUMMARY_PATH, 'model_{}'.format(num_folds))
    mkdirp(summary_path)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2, verbose=0, mode='auto'),
        ModelCheckpoint(checkpoint_path, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        TensorBoard(log_dir=summary_path, histogram_freq=0)
    ]
    model.fit(X_train, y_train, \
            batch_size=BATCH_SIZE, nb_epoch=NB_EPOCHS, \
            shuffle=True, \
            verbose=1, \
            validation_data=(X_valid, y_valid), \
            callbacks=callbacks)

    predictions_valid = model.predict(X_valid, batch_size=100, verbose=1)
    score_valid = log_loss(y_valid, predictions_valid)
    scores_total.append(score_valid)

    print('Score: {}'.format(score_valid))

    predictions_test = model.predict(X_test, batch_size=100, verbose=1)
    predictions_total.append(predictions_test)

    num_folds += 1

score_geom = calc_geom(scores_total, MAX_FOLDS)
predictions_geom = calc_geom_arr(predictions_total, MAX_FOLDS)

submission_path = os.path.join(SUMMARY_PATH, 'submission_{}_{:.2}.csv'.format(int(time.time()), score_geom))
write_submission(predictions_geom, X_test_ids, submission_path)
