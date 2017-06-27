import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv3D, ZeroPadding3D
from keras.layers.pooling import MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.data_utils import Sequence
from keras.layers.advanced_activations import PReLU

import numpy as np
from glob import glob
import pandas as pd
import os
import math

luna_path = "../inputs/CSVFILES/"
luna_subset_path = "../inputs/subset*/"

mhd_file_list = glob(luna_subset_path + "*.mhd")
resampled_file_list = glob("../preprocess/*.npz")

def load_annotations(csv_file):
    def get_filename(case):
        global mhd_file_list
        for f in mhd_file_list:
            if case in f:
                return(f)

    df_node = pd.read_csv(luna_path + "annotations.csv")
    df_node["file"] = df_node["seriesuid"].apply(get_filename)
    df_node = df_node.dropna()

    return df_node

def get_seriesuid(filepath):
    basename = os.path.basename(filepath)
    
    return basename[:-8]

def get_mhd_file(seriesuid):
    return get_filename(seriesuid)

def switch_xz(arr):
    return np.array([arr[2], arr[1], arr[0]])

def get_nodule_index(nodule_info, new_spacing, origin_zyx):
    nodule_coordinates = np.array(nodule_info[1:4]).astype(np.float32) # x,y,z

    nodule_index = np.rint((switch_xz(nodule_coordinates) - origin_zyx) / new_spacing).astype(np.int16)  # z,y,x
    
    return nodule_index

def get_random_index(shape, margin, cube_size, nodule_index_list):
    while True:
        random_index = [np.random.randint(margin, s - margin) for s in shape]

        if nodule_index_list is None or len(nodule_index_list) == 0:
            return random_index

        dist = min([np.linalg.norm(nodule_index - random_index) for nodule_index in nodule_index_list])

        if dist > math.sqrt(cube_size * cube_size * 2):
            return random_index
        else:
            # print("too close to nodule {}, nodule_list={}".format(random_index, nodule_index_list))
            continue
            
def get_range(x, minx, maxx, dim):
    hdim = dim // 2

    if x - hdim <= minx:
        xs = minx
        xe = xs + dim
    elif x + hdim >= maxx:
        xe = maxx
        xs = xe - dim
    else:
        xs = x - hdim
        xe = xs + dim
    
    return xs, xe

def get_3D_cube(img_array, x, y, z, dim):
    xr = get_range(x, 0, img_array.shape[2], dim)
    yr = get_range(y, 0, img_array.shape[1], dim)
    zr = get_range(z, 0, img_array.shape[0], dim)

    return img_array[zr[0]:zr[1], yr[0]:yr[1], xr[0]:xr[1]]

def get_feature_label_pairs(npz_file, count=2, cube_size=48, random_center=True, positive_portion=0.5):
    npz_dict = np.load(npz_file)
    resampled_img = npz_dict['resampled_img'].astype(np.float16) # zyx
    origin_zyx = npz_dict['origin_zyx']
    new_spacing = npz_dict['new_spacing'] #zyx
    seriesuid = get_seriesuid(npz_file)
    mini_df_node = df_node[df_node['seriesuid'] == seriesuid]

    features_list = []
    labels_list = []

    if len(mini_df_node) > 0:
        nodule_index_list = [get_nodule_index(nodule_info, new_spacing, origin_zyx) for nodule_info in mini_df_node.values] #zyx
    else:
        nodule_index_list = []
        positive_portion = 0.0

    pos_count = math.ceil(count * positive_portion)
    neg_count = count - pos_count

    # positive
    if len(mini_df_node) > 0:
        for i in range(pos_count):
            nodule_index = nodule_index_list[np.random.randint(len(nodule_index_list))]
            xi, yi, zi = nodule_index[2], nodule_index[1], nodule_index[0]

            # Randomize center
            if random_center:
                randi = np.random.randint(-5, +5, 3)
                xi += randi[0]
                yi += randi[1]
                zi += randi[2]

            nodule_cube = get_3D_cube(resampled_img, xi, yi, zi, cube_size)

            features_list.append(np.reshape(nodule_cube, [cube_size, cube_size, cube_size, 1]))
            labels_list.append(1)

    # negative
    for i in range(neg_count):
        random_index = get_random_index(resampled_img.shape, 0, cube_size, nodule_index_list)

        random_cube = get_3D_cube(resampled_img, random_index[2], random_index[1], random_index[0], cube_size)

        features_list.append(np.reshape(random_cube, [cube_size, cube_size, cube_size, 1]))
        labels_list.append(0)

    return features_list, labels_list

def add_conv_block(model, filter, kernel=(3, 3, 3), input_shape=None):
    model.add(Conv3D(filter, kernel, activation='linear', padding='valid', strides=(1, 1, 1), input_shape=input_shape))
    model.add(PReLU(shared_axes=[1,2,3]))
    model.add(BatchNormalization()) 

def get_model(input_dim=48):
    model = Sequential()
    # 1st layer group
    add_conv_block(model, 64, input_shape=(input_dim, input_dim, input_dim, 1))
    add_conv_block(model, 64)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid'))

    # 2nd layer group
    add_conv_block(model, 128)
    add_conv_block(model, 128)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid'))

    # 3rd layer group
    add_conv_block(model, 256)
    add_conv_block(model, 256)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid'))

    # 4th layer group
    add_conv_block(model, 512)
    add_conv_block(model, 512)
    model.add(MaxPooling3D(pool_size=(2, 2, 2), padding='valid'))

    model.add(Flatten())
    
    # FC layers group
    model.add(Dense(512, activation='linear'))
    model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation='linear'))
    model.add(PReLU(shared_axes=[1]))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True), metrics=['accuracy'])

    print(model.summary())

    return model

class DataSequence(Sequence):
    def __init__(self, file_list, cube_size, batch_size):
        self.file_list = file_list
        self.cube_size = cube_size
        self.batch_size = batch_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        npz_file = self.file_list[idx]

        fl, ll = get_feature_label_pairs(npz_file, count=self.batch_size, cube_size=self.cube_size, random_center=True)

        #import collections
        #print(idx, collections.Counter(ll), nl)

        return np.asarray(fl), np.asarray(ll)

from datetime import datetime as dt
def get_experiment_id():
    time_str = dt.now().strftime('%Y-%m-%d-%H-%M-%S')
    experiment_id = 'base_3d_cnn_48_{}'.format(time_str)

    return experiment_id

def train():
    resampled_file_list = glob("../preprocess/*.npz")
    n_val_files = 80
    val_file_list = resampled_file_list[:n_val_files]
    train_file_list = resampled_file_list[n_val_files:]

    input_dim = 48
    batch_size = 8

    train_gen = DataSequence(train_file_list, input_dim, batch_size)
    validate_gen = DataSequence(val_file_list, input_dim, batch_size)

    model = get_model(input_dim)

    experiment_id = get_experiment_id()

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        # ModelCheckpoint(experiment_id + "-{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        ModelCheckpoint(experiment_id + "-val_loss_checkpoint.hdf5", monitor='val_loss', verbose=1, save_best_only=True, mode='auto'),
        #ModelCheckpoint(experiment_id + "-val_categorical_accuracy_checkpoint.hdf5", monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='auto'),
        #keras.callbacks.TensorBoard(log_dir=experiment_id + "-tensorboard", histogram_freq=1, write_images=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=1e-6)
    ]
    
    history = model.fit_generator(generator=train_gen, 
                                  validation_data=validate_gen,
                                  steps_per_epoch=len(train_file_list),
                                  validation_steps=len(val_file_list) * 3,
                                  verbose=1,
                                  epochs=100,
                                  callbacks=callbacks,
                                  workers=7, # recommend: number of cpu cores
                                  use_multiprocessing=True)
    
    np.save(experiment_id + "_history.npy", history)

train()