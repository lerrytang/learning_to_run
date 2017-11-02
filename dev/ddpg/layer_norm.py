from keras.engine.topology import Layer
from keras import initializers
import keras.backend as K


class LayerNorm(Layer):
    """ Layer Normalization in the style of https://arxiv.org/abs/1607.06450 """

    def __init__(self, scale_initializer='ones', bias_initializer='zeros', **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.trainable = kwargs["trainable"]
        self.epsilon = 1e-8
        self.scale_initializer = initializers.get(scale_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

    def build(self, input_shape):
        self.scale = self.add_weight(shape=(),
                                     initializer=self.scale_initializer,
                                     trainable=self.trainable,
                                     name='{}_scale'.format(self.name))
        self.bias = self.add_weight(shape=(),
                                    initializer=self.bias_initializer,
                                    trainable=self.trainable,
                                    name='{}_bias'.format(self.name))
        super(LayerNorm, self).build(input_shape)

    def call(self, x, mask=None):
        mean = K.mean(x, axis=-1, keepdims=True)
        std = K.std(x, axis=-1, keepdims=True)
        norm = (x - mean) * (1.0 / (std + self.epsilon))
        return norm * self.scale + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape
