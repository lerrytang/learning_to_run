from keras.engine.topology import Layer
import keras.backend as K
import tensorflow as tf


class LayerSELU(Layer):

    def __init__(self, **kwargs):
        super(LayerSELU, self).__init__(**kwargs)

    def call(self, x, mask=None):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

    def compute_output_shape(self, input_shape):
        return input_shape
