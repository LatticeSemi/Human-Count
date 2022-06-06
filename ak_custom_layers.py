import sys
import os
import math

from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Lambda, Dropout, Flatten, Dense
from tensorflow.keras import Model, Input
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import TruncatedNormal
from typing import Union

from kerastuner.engine import hyperparameters
from tensorflow.keras import applications
from tensorflow.python.util import nest

from autokeras import keras_layers
from autokeras.engine import block as block_module
from autokeras.utils import layer_utils


from typing import Optional

import tensorflow as tf
from tensorflow.keras import activations
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.python.util import nest

from autokeras import adapters
from autokeras import analysers
from autokeras import hyper_preprocessors as hpps_module
from autokeras import preprocessors
from autokeras.blocks import reduction
from autokeras.engine import head as head_module
from autokeras.utils import types
from autokeras.utils import utils


from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import autokeras as ak
import math

import binary_ops
from binary_ops import *



class MyConstraints(tf.keras.constraints.Constraint):  ##Used for 8-bit weight quantization is Keras
    def __init__(self, name="", **kwargs):
        super(MyConstraints, self).__init__(**kwargs)
        self.name = name

    def __call__(self, w):
        with tf.compat.v1.variable_scope(self.name + "_CONSTRIANTS") as scope:
            return self.nested_lin_8b_quant(w)

    @staticmethod
    def nested_lin_8b_quant(w, min_rng=-0.5, max_rng=0.5):  ## 8-bit activation quantization in Keras using Lambda layer
        if min_rng == 0.0 and max_rng == 2.0:
            min_clip = 0
            max_clip = 255
        else:
            min_clip = -128
            max_clip = 127
        wq = 256.0 * w / (max_rng - min_rng)  # to expand [min, max] to [-128, 128]
        wq = K.round(wq)  # integer (quantization)
        wq = K.clip(wq, min_clip, max_clip)  # fit into 256 linear quantization
        wq = wq / 256.0 * (max_rng - min_rng)  # back to quantized real number, not integer
        wclip = K.clip(w, min_rng, max_rng)  # linear value w/ clipping
        return wclip + K.stop_gradient(wq - wclip)


class LatticeConvBlock(block_module.Block):
    """Block for vanilla ConvNets.

    # Arguments
        kernel_size: Int or keras_tuner.engine.hyperparameters.Choice.
            The size of the kernel.
            If left unspecified, it will be tuned automatically.
        num_blocks: Int or keras_tuner.engine.hyperparameters.Choice.
            The number of conv blocks, each of which may contain
            convolutional, max pooling, dropout, and activation. If left unspecified,
            it will be tuned automatically.
        num_layers: Int or hyperparameters.Choice.
            The number of convolutional layers in each block. If left
            unspecified, it will be tuned automatically.
        filters: Int or keras_tuner.engine.hyperparameters.Choice. The number of
            filters in the convolutional layers. If left unspecified, it will
            be tuned automatically.
        max_pooling: Boolean. Whether to use max pooling layer in each block. If left
            unspecified, it will be tuned automatically.
        separable: Boolean. Whether to use separable conv layers.
            If left unspecified, it will be tuned automatically.
        dropout: Float. Between 0 and 1. The dropout rate for after the
            convolutional layers. If left unspecified, it will be tuned
            automatically.
    """

    def __init__(
            self,
            kernel_size: Optional[Union[int, hyperparameters.Choice]] = None,
            num_blocks: Optional[Union[int, hyperparameters.Choice]] = None,
            num_layers: Optional[Union[int, hyperparameters.Choice]] = None,
            filters: Optional[Union[int, hyperparameters.Choice]] = None,
            max_pooling: Optional[bool] = None,
            separable: Optional[bool] = None,
            dropout: Optional[float] = None,
            use_batchnorm: Optional[bool] = None,
            quantrelu: Optional[bool] = None,
            kernel_quant: Optional[bool] = None,
            img_size: Optional[int] = None,
            **kwargs,
    ):
        super().__init__(**kwargs)
        min_layers = 2 * (int(math.log(img_size, 2)) - 2) + 1
        layer_choice = [i for i in range(min_layers, 26)]
        self.kernel_size = utils.get_hyperparameter(
            kernel_size,
            hyperparameters.Choice("kernel_size", [3], default=3),
            int,
        )
        self.num_blocks = utils.get_hyperparameter(
            num_blocks,
            hyperparameters.Choice("num_blocks", [1], default=1),
            int,
        )
        self.num_layers = utils.get_hyperparameter(
            num_layers,
            hyperparameters.Choice("num_layers", layer_choice, default=25),
            int,
        )
        self.filters = utils.get_hyperparameter(
            filters,
            hyperparameters.Choice(
                "filters", [4, 8, 16, 32, 64], default=32
            ),
            int,
        )
        self.max_pooling = max_pooling
        self.separable = separable
        self.dropout = dropout
        self.use_batchnorm = use_batchnorm
        self.quantrelu = quantrelu
        self.kernel_quant = kernel_quant

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "kernel_size": hyperparameters.serialize(self.kernel_size),
                "num_blocks": hyperparameters.serialize(self.num_blocks),
                "num_layers": hyperparameters.serialize(self.num_layers),
                "filters": hyperparameters.serialize(self.filters),
                "separable": self.separable,
                "dropout": self.dropout,
                "use_batchnorm": self.use_batchnorm,
                "quantrelu": self.quantrelu,
                "kernel_quant": self.kernel_quant
            }
        )
        return config

    @classmethod
    def from_config(cls, config):
        config["kernel_size"] = hyperparameters.deserialize(config["kernel_size"])
        config["num_blocks"] = hyperparameters.deserialize(config["num_blocks"])
        config["num_layers"] = hyperparameters.deserialize(config["num_layers"])
        config["filters"] = hyperparameters.deserialize(config["filters"])
        return cls(**config)

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        kernel_size = utils.add_to_hp(self.kernel_size, hp)
        normal_conv_kernel = kernel_size
        separable = False
        conv = tf.keras.layers.Conv2D

        pool = tf.keras.layers.MaxPool2D

        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0], default=0)
        use_batchnorm = self.use_batchnorm
        if use_batchnorm is None:
            use_batchnorm = hp.Boolean("use_batchnorm", default=True)
        max_pooling = False
        for i in range(utils.add_to_hp(self.num_blocks, hp)):
            for j in range(utils.add_to_hp(self.num_layers, hp)):
                max_pooling = not max_pooling
                if j == 1:
                    separable = self.separable
                    if separable is None:  # corner case when user doesn't provide separable value
                        separable = hp.Boolean("separable", default=False)
                    if separable:
                        conv1 = tf.keras.layers.DepthwiseConv2D
                        normal_conv_kernel = 1
                if output_node.shape[1] < kernel_size:
                    break
                if separable:
                    if self.kernel_quant:
                        output_node = conv1(
                            kernel_size=kernel_size,
                            padding="same", depthwise_initializer=TruncatedNormal(stddev=0.01, seed=99),
                            activation=None, use_bias=False,
                            depthwise_constraint=MyConstraints("depthwise" + str(i) + str(j)),
                            name="conv_DW_" + str(i) + "_" + str(j)
                        )(output_node)
                    else:
                        print("no quant in kernel")
                        output_node = conv1(
                            kernel_size=kernel_size,
                            padding="same", depthwise_initializer=TruncatedNormal(stddev=0.01, seed=99), \
                            activation=None, use_bias=False, name="conv_DW_" + str(i) + "_" + str(j)
                        )(output_node)
                    if use_batchnorm:
                        output_node = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-6,
                                                                         fused=False)(output_node)
                    if self.quantrelu:
                        output_node = Lambda(self._lin_8b_quant)(output_node)  ##Activation Quantization
                    output_node = tf.keras.layers.ReLU()(output_node)

                if self.kernel_quant:
                    output_node = conv(
                        utils.add_to_hp(
                            self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)
                        ),
                        normal_conv_kernel, padding="same", kernel_initializer=TruncatedNormal(stddev=0.01, seed=99), \
                        activation=None, use_bias=False, kernel_constraint=MyConstraints("pointwise" + str(i) + str(j)),
                        name="conv_PW_" + str(i) + "_" + str(j)
                    )(output_node)
                else:
                    print('NO QUANTIZATION IN KERNEL')
                    output_node = conv(
                        utils.add_to_hp(
                            self.filters, hp, "filters_{i}_{j}".format(i=i, j=j)
                        ),
                        normal_conv_kernel, padding="same", kernel_initializer=TruncatedNormal(stddev=0.01, seed=99), \
                        activation=None, use_bias=False, name="conv_PW_" + str(i) + "_" + str(j)
                    )(output_node)
                if use_batchnorm:
                    output_node = tf.keras.layers.BatchNormalization(axis=-1, momentum=0.9, epsilon=1e-6, fused=False)(
                        output_node)
                if self.quantrelu:
                    output_node = Lambda(self._lin_8b_quant)(output_node)  ##Activation Quantization
                output_node = tf.keras.layers.ReLU()(output_node)
                if max_pooling:
                    output_node = pool(pool_size=(2, 2), strides=2, padding='valid', data_format=None)(output_node)
            if dropout > 0:
                output_node = layers.Dropout(dropout)(output_node)
        return output_node

    @staticmethod
    def _get_padding(kernel_size, output_node):
        return "same"
        if all(kernel_size * 2 <= length for length in output_node.shape[1:-1]):
            return "valid"

    @staticmethod
    def _lin_8b_quant(w, min_rng=0.0, max_rng=2.0):  ## 8-bit activation quantization in Keras using Lambda layer
        if min_rng == 0.0 and max_rng == 2.0:
            min_clip = 0
            max_clip = 255
        else:
            min_clip = -128
            max_clip = 127
        wq = 256.0 * w / (max_rng - min_rng)  # to expand [min, max] to [-128, 128]
        wq = K.round(wq)  # integer (quantization)
        wq = K.clip(wq, min_clip, max_clip)  # fit into 256 linear quantization
        wq = wq / 256.0 * (max_rng - min_rng)  # back to quantized real number, not integer
        wclip = K.clip(w, min_rng, max_rng)  # linear value w/ clipping
        return wclip + K.stop_gradient(wq - wclip)


class LatClassificationHead(head_module.Head):
    """Classification Dense layers.

    Use sigmoid and binary crossentropy for binary classification and multi-label
    classification. Use softmax and categorical crossentropy for multi-class
    (more than 2) classification. Use Accuracy as metrics by default.

    The targets passing to the head would have to be tf.data.Dataset, np.ndarray,
    pd.DataFrame or pd.Series. It can be raw labels, one-hot encoded if more than two
    classes, or binary encoded for binary classification.

    The raw labels will be encoded to one column if two classes were found,
    or one-hot encoded if more than two classes were found.

    # Arguments
        num_classes: Int. Defaults to None. If None, it will be inferred from the
            data.
        multi_label: Boolean. Defaults to False.
        loss: A Keras loss function. Defaults to use `binary_crossentropy` or
            `categorical_crossentropy` based on the number of classes.
        metrics: A list of Keras metrics. Defaults to use 'accuracy'.
        dropout: Float. The dropout rate for the layers.
            If left unspecified, it will be tuned automatically.
    """

    def __init__(
            self,
            num_classes: Optional[int] = None,
            multi_label: bool = False,
            loss: Optional[types.LossType] = None,
            metrics: Optional[types.MetricsType] = None,
            dropout: Optional[float] = None,
            kernel_quant: Optional[bool] = None,
            **kwargs
    ):
        self.num_classes = num_classes
        self.multi_label = multi_label
        self.dropout = dropout
        if metrics is None:
            metrics = ["accuracy"]
        if loss is None:
            loss = self.infer_loss()
        super().__init__(loss=loss, metrics=metrics, **kwargs)
        # Infered from analyser.
        self._encoded = None
        self._encoded_for_sigmoid = None
        self._encoded_for_softmax = None
        self._add_one_dimension = False
        self._labels = None
        self.kernel_quant = kernel_quant

    def infer_loss(self):
        if not self.num_classes:
            return None
        if self.num_classes == 2 or self.multi_label:
            return losses.BinaryCrossentropy()
        return losses.CategoricalCrossentropy()

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "num_classes": self.num_classes,
                "multi_label": self.multi_label,
                "dropout": self.dropout,
                "kernel_quant": self.kernel_quant
            }
        )
        return config

    def build(self, hp, inputs=None):
        inputs = nest.flatten(inputs)
        utils.validate_num_inputs(inputs, 1)
        input_node = inputs[0]
        output_node = input_node

        output_node = tf.keras.layers.Flatten()(output_node)
        if self.dropout is not None:
            dropout = self.dropout
        else:
            dropout = hp.Choice("dropout", [0.0, 0.25, 0.5], default=0)

        if dropout > 0:
            output_node = layers.Dropout(dropout)(output_node)
        if self.kernel_quant:
            output_node = layers.Dense(self.shape[-1], kernel_constraint=MyConstraints("Dense"),
                                       kernel_initializer=tf.keras.initializers.GlorotNormal(seed=99),
                                       bias_initializer="zeros", use_bias=True)(output_node)
        else:
            print('NO QUANTIZATION IN KERNEL')
            output_node = layers.Dense(self.shape[-1], kernel_initializer=tf.keras.initializers.GlorotNormal(seed=99),
                                       bias_initializer="zeros", use_bias=True)(output_node)

        if isinstance(self.loss, tf.keras.losses.BinaryCrossentropy):
            output_node = layers.Activation(activations.sigmoid, name=self.name)(
                output_node
            )
        else:
            output_node = layers.Softmax(name=self.name)(output_node)
        return output_node

    def get_adapter(self):
        return adapters.ClassificationAdapter(name=self.name)

    def get_analyser(self):
        return analysers.ClassificationAnalyser(
            name=self.name, multi_label=self.multi_label
        )

    def config_from_analyser(self, analyser):
        super().config_from_analyser(analyser)
        self.num_classes = analyser.num_classes
        self.loss = self.infer_loss()
        self._encoded = analyser.encoded
        self._encoded_for_sigmoid = analyser.encoded_for_sigmoid
        self._encoded_for_softmax = analyser.encoded_for_softmax
        self._add_one_dimension = len(analyser.shape) == 1
        self._labels = analyser.labels

    def get_hyper_preprocessors(self):
        hyper_preprocessors = []

        if self._add_one_dimension:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.AddOneDimension())
            )

        if self.dtype in [tf.uint8, tf.uint16, tf.uint32, tf.uint64]:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToInt32())
            )

        if not self._encoded and self.dtype != tf.string:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(preprocessors.CastToString())
            )

        if self._encoded_for_sigmoid:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SigmoidPostprocessor()
                )
            )
        elif self._encoded_for_softmax:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.SoftmaxPostprocessor()
                )
            )
        elif self.num_classes == 2:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.LabelEncoder(self._labels)
                )
            )
        else:
            hyper_preprocessors.append(
                hpps_module.DefaultHyperPreprocessor(
                    preprocessors.OneHotEncoder(self._labels)
                )
            )
        return hyper_preprocessors

