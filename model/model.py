import sys

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.constraints import MinMaxNorm
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Lambda

import model.binary_ops as bo


def bin_wrapper_layer(x, a_bin=16, min_rng=0.0, max_rng=2.0):  # activation binarization
    x_quant = bo.lin_8b_quant(x, min_rng=min_rng, max_rng=max_rng)
    return x_quant


def bin_wrapper_layer_last_layer(x, a_bin=16, min_rng=-32.0, max_rng=+32.0):  # activation binarization
    x_quant = bo.lin_8b_quant(x, min_rng=min_rng, max_rng=max_rng)
    return x_quant


class NormDense(tf.keras.layers.Layer):
    def __init__(self, units=1000, kernel_regularizer=None, loss_top_k=1, **kwargs):
        super(NormDense, self).__init__(**kwargs)
        self.init = tf.keras.initializers.glorot_normal()
        self.units, self.loss_top_k = units, loss_top_k
        self.kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self.supports_masking = False

    def build(self, input_shape):
        self.w = self.add_weight(
            name="norm_dense_w",
            shape=(input_shape[-1], self.units * self.loss_top_k),
            initializer=self.init,
            trainable=True,
            regularizer=self.kernel_regularizer,
        )
        super(NormDense, self).build(input_shape)

    def call(self, inputs, **kwargs):
        norm_w = K.l2_normalize(self.w, axis=0)
        inputs = K.l2_normalize(inputs, axis=1)
        output = K.dot(inputs, norm_w)
        if self.loss_top_k > 1:
            output = K.reshape(output, (-1, self.units, self.loss_top_k))
            output = K.max(output, axis=2)
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.units)

    def get_config(self):
        config = super(NormDense, self).get_config()
        config.update(
            {
                "units": self.units,
                "loss_top_k": self.loss_top_k,
                "kernel_regularizer": tf.keras.regularizers.serialize(self.kernel_regularizer),
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


# class that wraps config and model
class FaceRecognition:
    # initialize model from config file
    def __init__(self, config):
        """Init of SqueezeDet Class

        Arguments:
            config {[type]} -- dict containing hyperparameters for network building
        """

        # hyperparameter config file
        self.config = config
        self.depth = config.FILTER_DEPTHS
        self.useconv3 = config.USE_CONV3
        depth_length = len(self.depth)
        self.weight_decay = 1e-4

        if depth_length < 6 or depth_length > 12:
            print("Length of Depths Should be in between 6-10")
            sys.exit()
        if config.EARLY_POOLING:
            if depth_length == 6:
                self.pooling = [True, True, True, False, True, False]
            elif depth_length == 7:
                self.pooling = [True, False, True, False, True, False, True]
            elif depth_length == 8:
                self.pooling = [True, False, True, False, True, False, True, False]
            elif depth_length == 9:
                self.pooling = [True, False, True, True, False, False, True, False, False]
            elif depth_length == 12:
                self.pooling = [True, False, False, True, False, True, False, True, False, False, False, False]
        else:
            if depth_length == 6:
                self.pooling = [True, False, True, True, True, False]
            elif depth_length == 7:
                self.pooling = [True, False, True, False, True, False, True]
            elif depth_length == 8:
                self.pooling = [True, False, True, False, True, False, True, False]
            elif depth_length == 9:
                self.pooling = [True, False, True, False, True, False, True, False, False]
            else:
                self.pooling = [True, False, False, True, False, True, False, True, False, False]

        # create Keras model
        self.model = self._create_model()

    def bin_wrapper_layer(self, x):  # activation binarization
        x_quant = bo.lin_8b_quant(x, min_rng=self.config.QUANT_RANGE[0], max_rng=self.config.QUANT_RANGE[1])
        return x_quant

    def _create_model(self):
        """
        #builds the Keras model from config
        #return: squeezeDet in Keras
        """
        self.input_layer = Input(shape=(self.config.IMAGE_HEIGHT, self.config.IMAGE_WIDTH, self.config.N_CHANNELS),
                                 name="input")
        prev_layer = self.input_layer
        if self.useconv3:
            self.fire1 = self._fire_layer(name="fire1", input=prev_layer, out_channels=self.depth[0], stdd=0.012,
                                          wt_quant=True, act_quant=True, bn_eps=1e-3, pool=self.pooling[0])
        else:
            self.fire1 = self._mobile_layer(name="fire1", input=prev_layer, out_channels=self.depth[0], stdd=0.012,
                                            wt_quant=True, act_quant=True, bn_eps=1e-3, pool=self.pooling[0],
                                            pool2=True)
        prev_layer = self.fire1
        for layer_count in range(len(self.depth) - 1):
            if self.config.BACK_BONE == "VGG":
                prev_layer = self._fire_layer(name="fire{}".format(str(layer_count + 2)), input=prev_layer,
                                              out_channels=self.depth[layer_count + 1],
                                              stdd=0.012, wt_quant=True, act_quant=True,
                                              bn_eps=1e-3, pool=self.pooling[layer_count + 1])
            elif self.config.BACK_BONE == "MV1":
                prev_layer = self._mobile_layer(name="fire{}".format(str(layer_count + 2)), input=prev_layer,
                                                out_channels=self.depth[layer_count + 1],
                                                stdd=0.012, wt_quant=True, act_quant=True,
                                                bn_eps=1e-3, pool=self.pooling[layer_count + 1])
        dropout = 0.3
        use_bias = True
        if 0 < dropout < 1:
            nn = Dropout(dropout)(prev_layer)
        else:
            nn = prev_layer
        nn = Flatten(name="E_flatten")(nn)
        nn = Dense(self.config.FEATURES, use_bias=use_bias, kernel_initializer="glorot_normal",
                   name="E_dense", kernel_constraint=MinMaxNorm(
                min_value=-0.5, max_value=0.5, rate=1.0, axis=0))(nn)

        nn = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(nn)

        return Model(self.input_layer, nn)

    @staticmethod
    def _fire_layer(name, input, out_channels, stdd=0.01, wt_quant=True, act_quant=True, bn_eps=1e-6, pool=True):
        """
            wrapper for fire layer constructions

            :param name: name for layer
            :param input: previous layer
            :param s1x1: number of filters for squeezing
            :param e1x1: number of filter for expand 1x1
            :param e3x3: number of filter for expand 3x3
            :param stdd: standard deviation used for intialization
            :return: a keras fire layer
            """
        with tf.compat.v1.variable_scope(name) as scope:
            ex3x3 = Conv2D(
                name=name + '/expand3x3', filters=out_channels, kernel_size=(3, 3), strides=(1, 1), use_bias=False,
                padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd),
                kernel_constraint=bo.MyConstraints("quant_" + name))(input)
            bch1 = BatchNormalization(epsilon=bn_eps, trainable=True)
            ex3x3 = bch1(ex3x3)

            if act_quant:
                ex3x3 = Lambda(bin_wrapper_layer)(ex3x3)
            ex3x3 = ReLU()(ex3x3)
            if pool:
                ex3x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=name + "fl_pool")(ex3x3)

        return ex3x3

    # wrapper for padding, written in tensorflow. If you want to change to theano you need to rewrite this!
    @staticmethod
    def _pad(input):
        """
        pads the network output so y_pred and y_true have the same dimensions
        :param input: previous layer
        :return: layer, last dimensions padded for 4
        """

        padding = np.zeros((3, 2))
        padding[2, 1] = 4
        return tf.pad(tensor=input, paddings=padding, mode="CONSTANT")

    @staticmethod
    def _mobile_layer(name, input, out_channels=1, stdd=0.02, wt_quant=True, act_quant=True, bn_eps=1e-6,
                      pool=True, pool2=False):
        with tf.compat.v1.variable_scope(name + "_DW") as scope:
            ex3x3 = DepthwiseConv2D(name=name + "_DW", kernel_size=3, padding="SAME", use_bias=False,
                                    depthwise_initializer=TruncatedNormal(stddev=stdd),
                                    depthwise_constraint=bo.MyConstraints("quant_" + name + "_DW"))(input)

            bch2 = BatchNormalization(epsilon=bn_eps, trainable=True)
            ex3x3 = bch2(ex3x3)

            if act_quant:
                ex3x3 = Lambda(bin_wrapper_layer)(ex3x3)
            ex3x3 = ReLU()(ex3x3)
            if pool:
                ex3x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=name + "ml_pool")(ex3x3)
        with tf.compat.v1.variable_scope(name + "_PW") as scope:
            ex3x3 = Conv2D(
                name=name + '_PW', filters=out_channels, kernel_size=(1, 1), strides=(1, 1), use_bias=False,
                padding='SAME', kernel_initializer=TruncatedNormal(stddev=stdd),
                kernel_constraint=bo.MyConstraints("quant_" + name + "_PW"))(ex3x3)

            bch3 = BatchNormalization(epsilon=bn_eps, trainable=True)
            ex3x3 = bch3(ex3x3)

            if act_quant:
                ex3x3 = Lambda(bin_wrapper_layer)(ex3x3)
            ex3x3 = ReLU()(ex3x3)
            if pool2:
                ex3x3 = MaxPool2D(pool_size=(2, 2), strides=(2, 2), name=name + "ml_pool_2")(ex3x3)
            return ex3x3
