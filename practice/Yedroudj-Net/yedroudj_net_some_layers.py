import numpy as np
import tensorflow as tf
from keras import layers

class ABS(layers.Layer):
    def call(self, inputs):
        return tf.abs(inputs)



class HPF(layers.Layer):
    def __init__(self, all_normalized_hpf_list, T=3.0, **kwargs):
        super().__init__(**kwargs)
        hpf_list_5x5 = []
        for h in all_normalized_hpf_list:
            h = np.asarray(h)
            if h.shape == (3,3):
                h = np.pad(h, pad_width=((1,1),(1,1)), mode='constant')
            elif h.shape == (5,5):
                pass
            else:
                raise ValueError("each kernel must be 3x3 or 5x5")
            hpf_list_5x5.append(h)
        kernels = np.stack(hpf_list_5x5, axis=0)
        kernels = kernels[:,:,:,np.newaxis]
        kernels = np.transpose(kernels, (1,2,3,0))
        self._init_kernels = kernels.astype(np.float32)
        self.conv = layers.Conv2D(filters=self._init_kernels.shape[-1],
                                  kernel_size=5, strides=1, padding='same',
                                  use_bias=False, trainable=False)
        self.tlu = Trunc(scale=T)
    def build(self, input_shape):
        self.conv.build(input_shape)
        self.conv.set_weights([self._init_kernels])
        self.conv.trainable = False
        super().build(input_shape)
    def call(self, inputs):
        return self.tlu(self.conv(inputs))



class Scale(layers.Layer):
    def __init__(self, axis=-1, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.epsilon = epsilon
    def build(self, input_shape):
        ch = int(input_shape[self.axis])
        self.gamma = self.add_weight(name='gamma', shape=(ch,), initializer=tf.keras.initializers.Ones(), trainable=True)
        self.beta = self.add_weight(name='beta', shape=(ch,), initializer='zeros', trainable=True)
        super().build(input_shape)
    def call(self, x):
        return x * tf.reshape(self.gamma, [1,1,1,-1]) + tf.reshape(self.beta, [1,1,1,-1])



class Trunc(layers.Layer):
    def __init__(self, scale=3.0, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale
    def call(self, inputs):
        return tf.clip_by_value(inputs, -self.scale, self.scale)
