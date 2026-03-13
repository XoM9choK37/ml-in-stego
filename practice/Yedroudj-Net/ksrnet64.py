import tensorflow as tf
from keras import models, regularizers, layers

from yedroudj_net import Scale, HPF, ABS, Trunc

# 1, 2, 4, new_2, new
def ksrnet64(input_shape, all_normalized_hpf_list):
    inp = layers.Input(shape=input_shape)



    x0 = HPF(all_normalized_hpf_list, T=3.0)(inp)



    x = layers.Conv2D(16, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv1',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x0)
    x = ABS()(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1')(x)
    x = Scale(name='scale1')(x)
    x = Trunc(scale=3.0, name='trunc1')(x)
    
    
    
    x0 = layers.Conv2D(16, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv1_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x0)
    x0 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1_par')(x0)
    x0 = Scale(name='scale1_par')(x0)

    x1 = tf.keras.layers.Concatenate()([x, x0])
    
    
    
    x = layers.Conv2D(16, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv2',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x1)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2')(x)
    x = Scale(name='scale2')(x)
    x = Trunc(scale=2.0, name='trunc2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool2')(x)



    x1 = layers.Conv2D(16, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv2_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x1)
    x1 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2_par')(x1)
    x1 = Scale(name='scale2_par')(x1)

    x2 = tf.keras.layers.Concatenate()([x, x1])



    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv3',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x2)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3')(x)
    x = Scale(name='scale3')(x)
    x = layers.ELU(name='elu1')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool3')(x)



    x2 = layers.Conv2D(16, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv3_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x2)
    x2 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3_par')(x2)
    x2 = Scale(name='scale3_par')(x2)

    x3 = tf.keras.layers.Concatenate()([x, x2])



    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv4',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x3)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4')(x)
    x = Scale(name='scale4')(x)
    x = layers.ELU(name='elu2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool4')(x)



    x3 = layers.Conv2D(32, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv4_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x3)
    x3 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4_par')(x3)
    x3 = Scale(name='scale4_par')(x3)
    
    x4 = tf.keras.layers.Concatenate()([x, x3])



    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv5',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x4)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5')(x)
    x = Scale(name='scale5')(x)
    x = layers.ELU(name='elu3')(x)
    
    

    x4 = layers.Conv2D(64, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x4)
    x4 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5_par')(x4)
    x4 = Scale(name='scale5_par')(x4)

    x5 = tf.keras.layers.Concatenate()([x, x4])



    x = layers.GlobalAveragePooling2D(name='pool5')(x5)

    x = layers.Dense(256, activation='elu', name='fc1',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(64, activation='elu', name='fc2',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2, name='fc3',
                     kernel_initializer="glorot_uniform")(x)
    out = layers.Softmax(axis=1, name='prob')(x)

    model = models.Model(inputs=inp, outputs=out, name='ksrnet64')
    return model



# new_1
"""def ksrnet64(input_shape, all_normalized_hpf_list):
    inp = layers.Input(shape=input_shape)



    x0 = HPF(all_normalized_hpf_list, T=3.0)(inp)



    x = layers.Conv2D(16, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv1',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x0)
    x = ABS()(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1')(x)
    x = Scale(name='scale1')(x)
    x = Trunc(scale=3.0, name='trunc1')(x)
    
    
    
    x0 = layers.Conv2D(16, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv1_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x0)
    x0 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1_par')(x0)
    x0 = Scale(name='scale1_par')(x0)

    x1 = tf.keras.layers.Concatenate()([x, x0])
    
    
    
    x = layers.Conv2D(16, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv2',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x1)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2')(x)
    x = Scale(name='scale2')(x)
    x = Trunc(scale=2.0, name='trunc2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool2')(x)



    x1 = layers.Conv2D(16, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv2_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x1)
    x1 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2_par')(x1)
    x1 = Scale(name='scale2_par')(x1)

    x2 = tf.keras.layers.Concatenate()([x, x1])



    x = layers.Conv2D(16, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv3',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x2)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3')(x)
    x = Scale(name='scale3')(x)
    x = layers.ReLU(name='relu1')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool3')(x)



    x2 = layers.Conv2D(16, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv3_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x2)
    x2 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3_par')(x2)
    x2 = Scale(name='scale3_par')(x2)

    x3 = tf.keras.layers.Concatenate()([x, x2])



    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv4',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x3)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4')(x)
    x = Scale(name='scale4')(x)
    x = layers.ReLU(name='relu2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool4')(x)



    x3 = layers.Conv2D(32, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv4_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x3)
    x3 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4_par')(x3)
    x3 = Scale(name='scale4_par')(x3)
    
    x4 = tf.keras.layers.Concatenate()([x, x3])



    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv5',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x4)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5')(x)
    x = Scale(name='scale5')(x)
    x = layers.ReLU(name='relu3')(x)
    
    

    x4 = layers.Conv2D(64, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x4)
    x4 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5_par')(x4)
    x4 = Scale(name='scale5_par')(x4)

    x5 = tf.keras.layers.Concatenate()([x, x4])



    x = layers.GlobalAveragePooling2D(name='pool5')(x5)

    x = layers.Dense(256, activation='relu', name='fc1',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(64, activation='relu', name='fc2',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2, name='fc3',
                     kernel_initializer="glorot_uniform")(x)
    out = layers.Softmax(axis=1, name='prob')(x)

    model = models.Model(inputs=inp, outputs=out, name='ksrnet64')
    return model"""



# 5
"""def ksrnet64(input_shape, all_normalized_hpf_list):
    inp = layers.Input(shape=input_shape)



    x0 = HPF(all_normalized_hpf_list, T=3.0)(inp)



    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv1',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x0)
    x = ABS()(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1')(x)
    x = Scale(name='scale1')(x)
    x = Trunc(scale=3.0, name='trunc1')(x)
    
    
    
    x0 = layers.Conv2D(30, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv1_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x0)
    x0 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1_par')(x0)
    x0 = Scale(name='scale1_par')(x0)

    x1 = tf.keras.layers.Concatenate()([x, x0])
    
    
    
    x = layers.Conv2D(30, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv2',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x1)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2')(x)
    x = Scale(name='scale2')(x)
    x = Trunc(scale=2.0, name='trunc2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool2')(x)



    x1 = layers.Conv2D(30, kernel_size=2, strides=2, padding='same', use_bias=False, name='conv2_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x1)
    x1 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2_par')(x1)
    x1 = Scale(name='scale2_par')(x1)

    x2 = tf.keras.layers.Concatenate()([x, x1])



    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv3',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x2)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3')(x)
    x = Scale(name='scale3')(x)
    x = layers.ELU(name='elu1')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool3')(x)



    x2 = layers.Conv2D(32, kernel_size=2, strides=2, padding='same', use_bias=False, name='conv3_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x2)
    x2 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3_par')(x2)
    x2 = Scale(name='scale3_par')(x2)

    x3 = tf.keras.layers.Concatenate()([x, x2])



    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv4',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x3)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4')(x)
    x = Scale(name='scale4')(x)
    x = layers.ELU(name='elu2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool4')(x)



    x3 = layers.Conv2D(64, kernel_size=2, strides=2, padding='same', use_bias=False, name='conv4_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x3)
    x3 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4_par')(x3)
    x3 = Scale(name='scale4_par')(x3)

    x4 = tf.keras.layers.Concatenate()([x, x3])



    x = layers.Conv2D(128, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv5',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x4)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5')(x)
    x = Scale(name='scale5')(x)
    x = layers.ELU(name='elu3')(x)
    
    

    x4 = layers.Conv2D(128, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x4)
    x4 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5_par')(x4)
    x4 = Scale(name='scale5_par')(x4)

    x5 = tf.keras.layers.Concatenate()([x, x4])



    x = layers.GlobalAveragePooling2D(name='pool5')(x5)

    x = layers.Dense(256, activation='elu', name='fc1',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(64, activation='elu', name='fc2',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2, name='fc3',
                     kernel_initializer="glorot_uniform")(x)
    out = layers.Softmax(axis=1, name='prob')(x)

    model = models.Model(inputs=inp, outputs=out, name='ksrnet64')
    return model"""



# 3
"""def ksrnet64(input_shape, all_normalized_hpf_list):
    inp = layers.Input(shape=input_shape)



    x0 = HPF(all_normalized_hpf_list, T=3.0)(inp)



    x = layers.Conv2D(16, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv1',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x0)
    x = ABS()(x)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1')(x)
    x = Scale(name='scale1')(x)
    x = Trunc(scale=3.0, name='trunc1')(x)
    
    
    
    x0 = layers.Conv2D(16, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv1_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x0)
    x0 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn1_par')(x0)
    x0 = Scale(name='scale1_par')(x0)

    x1 = tf.keras.layers.Concatenate()([x, x0])
    
    
    
    x = layers.Conv2D(20, kernel_size=5, strides=1, padding='same', use_bias=False, name='conv2',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x1)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2')(x)
    x = Scale(name='scale2')(x)
    x = Trunc(scale=2.0, name='trunc2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool2')(x)



    x1 = layers.Conv2D(20, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv2_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x1)
    x1 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn2_par')(x1)
    x1 = Scale(name='scale2_par')(x1)

    x2 = tf.keras.layers.Concatenate()([x, x1])



    x = layers.Conv2D(24, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv3',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x2)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3')(x)
    x = Scale(name='scale3')(x)
    x = layers.ELU(name='elu1')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool3')(x)



    x2 = layers.Conv2D(24, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv3_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x2)
    x2 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn3_par')(x2)
    x2 = Scale(name='scale3_par')(x2)

    x3 = tf.keras.layers.Concatenate()([x, x2])



    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv4',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x3)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4')(x)
    x = Scale(name='scale4')(x)
    x = layers.ELU(name='elu2')(x)

    x = layers.AveragePooling2D(pool_size=5, strides=2, padding='same', name='pool4')(x)



    x3 = layers.Conv2D(32, kernel_size=1, strides=2, padding='same', use_bias=False, name='conv4_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x3)
    x3 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn4_par')(x3)
    x3 = Scale(name='scale4_par')(x3)
    
    x4 = tf.keras.layers.Concatenate()([x, x3])



    x = layers.Conv2D(64, kernel_size=3, strides=1, padding='same', use_bias=False, name='conv5',
                      kernel_regularizer=regularizers.l2(0.0001),
                      kernel_initializer="glorot_uniform")(x4)
    x = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5')(x)
    x = Scale(name='scale5')(x)
    x = layers.ELU(name='elu3')(x)
    
    

    x4 = layers.Conv2D(64, kernel_size=1, strides=1, padding='same', use_bias=False, name='conv5_par',
                       kernel_regularizer=regularizers.l2(0.0001),
                       kernel_initializer="glorot_uniform")(x4)
    x4 = layers.BatchNormalization(momentum=0.001, epsilon=1e-5, center=False, scale=False, name='bn5_par')(x4)
    x4 = Scale(name='scale5_par')(x4)

    x5 = tf.keras.layers.Concatenate()([x, x4])



    x = layers.GlobalAveragePooling2D(name='pool5')(x5)

    x = layers.Dense(256, activation='elu', name='fc1',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(64, activation='elu', name='fc2',
                     kernel_regularizer=regularizers.l2(0.0001),
                     kernel_initializer="glorot_uniform")(x)
    x = layers.Dense(2, name='fc3',
                     kernel_initializer="glorot_uniform")(x)
    out = layers.Softmax(axis=1, name='prob')(x)

    model = models.Model(inputs=inp, outputs=out, name='ksrnet64')
    return model"""
