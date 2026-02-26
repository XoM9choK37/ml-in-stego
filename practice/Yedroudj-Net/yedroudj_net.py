import numpy as np
import tensorflow as tf
from keras import layers, regularizers, models

from yedroudj_net_some_layers import *

def yedroudj_net(input_shape, all_normalized_hpf_list):
    inp = layers.Input(shape=input_shape)

    x = HPF(all_normalized_hpf_list, T=3.0)(inp)

    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv1',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x)
    x = ABS()(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1')(x)
    x = Scale(name='scale1')(x)
    x = Trunc(scale=3.0, name='trunc1')(x)
    
    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv2',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2')(x)
    x = Scale(name='scale2')(x)
    x = Trunc(scale=2.0, name='trunc2')(x)

    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='valid', name='pool2')(x)

    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv3',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3')(x)
    x = Scale(name='scale3')(x)
    x = layers.ReLU(name='relu1')(x)

    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='valid', name='pool3')(x)

    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv4',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4')(x)
    x = Scale(name='scale4')(x)
    x = layers.ReLU(name='relu2')(x)

    x = layers.ZeroPadding2D(padding=1)(x)
    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='valid', name='pool4')(x)

    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv5',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5')(x)
    x = Scale(name='scale5')(x)
    x = layers.ReLU(name='relu3')(x)

    x = layers.GlobalAveragePooling2D(name='pool5')(x)

    x = layers.Dense(256, activation='relu', name='fc1',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(1024, activation='relu', name='fc2',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2, name='fc3',
                     kernel_initializer="glorot_uniform")(x)
    out = layers.Softmax(axis=1, name='prob')(x)

    model = models.Model(inputs=inp, outputs=out, name='yedroudj_net')
    return model
