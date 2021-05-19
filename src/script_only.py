# Import Modules

import math
import json
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from scipy import io
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *
from tqdm import tqdm 

%matplotlib inline

# Load data

train_data = np.load('train_data.npy')
test_data=np.load('test_data.npy')

# Data pre-processing

IMG_SIZE=75
X_train_orig=np.empty([len(train_data),IMG_SIZE,IMG_SIZE,2])
X_train_orig_lin=np.empty([len(train_data),IMG_SIZE,IMG_SIZE,2])
Y_train_orig=np.empty([len(train_data),1])
X_test_orig=np.empty([len(test_data),IMG_SIZE,IMG_SIZE,2])


for i in range(len(train_data)):
    X_train_orig[i,:,:,0]=np.asarray(train_data[i]['band_1']).reshape(IMG_SIZE,IMG_SIZE)
    X_train_orig[i,:,:,1]=np.asarray(train_data[i]['band_2']).reshape(IMG_SIZE,IMG_SIZE)
    Y_train_orig[i,0]=np.asarray(train_data[i]['is_iceberg']).reshape(1,1)
    X_train_orig_lin[i,:,:,0]=np.power(10,X_train_orig[i,:,:,0]/10)
    X_train_orig_lin[i,:,:,1]=np.power(10,X_train_orig[i,:,:,1]/10)
    if train_data[i]['inc_angle']!='na':
        if np.asarray(train_data[i]['inc_angle'])<30:
            mean=np.mean(X_train_orig_lin[i,:,:,1])
            std_dev=np.std(X_train_orig_lin[i,:,:,1])
            threshold=mean+0.8*std_dev
            ind_mat=X_train_orig_lin[i,:,:,1]>threshold
        else:
            mean=np.mean(X_train_orig_lin[i,:,:,0])
            std_dev=np.std(X_train_orig_lin[i,:,:,0])
            threshold=mean+0.8*std_dev
            ind_mat=X_train_orig_lin[i,:,:,0]>threshold
    else:
        mean=np.mean(X_train_orig_lin[i,:,:,0])
        std_dev=np.std(X_train_orig_lin[i,:,:,0])
        threshold=mean+0.8*std_dev
        ind_mat=X_train_orig_lin[i,:,:,0]>threshold
            
    X_train_orig[i,:,:,0]=np.multiply(X_train_orig[i,:,:,0],ind_mat)
    X_train_orig[i,:,:,1]=np.multiply(X_train_orig[i,:,:,1],ind_mat)
    
Y_train_orig=np.int_(Y_train_orig.reshape(1604,1))

for i in range(len(test_data)):
    X_test_orig[i,:,:,0]=np.asarray(test_data[i]['band_1']).reshape(IMG_SIZE,IMG_SIZE)
    X_test_orig[i,:,:,1]=np.asarray(test_data[i]['band_2']).reshape(IMG_SIZE,IMG_SIZE)

a=np.arange(X_train_orig.shape[0])

np.random.shuffle(a)
print(a.shape[0])
train_ind=1400
X_train=X_train_orig[a[0:train_ind]]
Y_train=Y_train_orig[a[0:train_ind]]


X_test=X_train_orig[a[train_ind:1604]]
Y_test=Y_train_orig[a[train_ind:1604]]

Y_train = convert_to_one_hot(Y_train, 2).T
Y_test = convert_to_one_hot(Y_test, 2).T

# Training the neural network

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten, one_hot_encoding
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation


# Create extra synthetic training data by flipping, rotating and blurring the
# images on our data set.
img_aug = ImageAugmentation()
img_aug.add_random_flip_leftright()
img_aug.add_random_rotation(max_angle=25.)
img_aug.add_random_blur(sigma_max=3.)

tf.reset_default_graph()
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 2], data_augmentation=img_aug, name='input')

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 3, padding ='valid')

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 3)

convnet = conv_2d(convnet, 128, 3, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 64, 3, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = conv_2d(convnet, 32, 3, activation='relu')
convnet = max_pool_2d(convnet, 5)

convnet = flatten(convnet)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.9)

convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam', learning_rate=1e-3, loss='categorical_crossentropy',name='targets')

model = tflearn.DNN(convnet, tensorboard_dir='log')

model.fit({'input': X_train}, {'targets': Y_train}, n_epoch=50, validation_set=({'input': X_test}, {'targets': Y_test}), 
    snapshot_step=500, show_metric=True)
    
# Model Prediction
model_out = model.predict(X_test_orig)