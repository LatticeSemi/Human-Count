# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import joblib
import numpy as np
import os
import sys
import tensorflow as tf
from easydict import EasyDict as edict
from nn_skeleton import ModelSkeleton
from utils import util


class SqueezeDet(ModelSkeleton):
    def __init__(self, mc, gpu_id=0):
        with tf.device('/gpu:{}'.format(gpu_id)):
            ModelSkeleton.__init__(self, mc)

            self._add_forward_graph()
            self._add_interpretation_graph()
            self._add_loss_graph()
            self._add_train_graph()
            self._add_viz_graph()

    def _add_forward_graph(self):
        """NN architecture."""

        mc = self.mc
        bin_k = 1  # K for BNN

        if mc.LOAD_PRETRAINED_MODEL:
            assert tf.gfile.Exists(mc.PRETRAINED_MODEL_PATH), \
                'Cannot find pretrained model at the given path:' \
                '  {}'.format(mc.PRETRAINED_MODEL_PATH)
            self.caffemodel_weight = joblib.load(mc.PRETRAINED_MODEL_PATH)

        fl_w_bin = 8
        fl_a_bin = 8
        ml_w_bin = 8
        ml_a_bin = 8
        ll_w_bin = 8
        ll_a_bin = 16

        min_rng = 0.0
        max_rng = 2.0

        bias_on = False

        depth = [8, 8, 8, 8, 40, 40, 40, 40, 80, 80, 80, 80, 160, 160, 160, 320, 320, 320]  # another thick version #12

        mul_f = 1

        fire1 = self._fire_layer('fire1', self.image_input, oc=depth[0], freeze=False, w_bin=fl_w_bin, a_bin=fl_a_bin,
                                 pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire2 = self._bsconv_layer('fire2', fire1, oc=depth[1], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire3 = self._bsconv_layer('fire3', fire2, oc=depth[2], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire4 = self._bsconv_layer('fire4', fire3, oc=depth[3], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire5 = self._bsconv_layer('fire5', fire4, oc=depth[4], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire6 = self._bsconv_layer('fire6', fire5, oc=depth[5], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire7 = self._bsconv_layer('fire7', fire6, oc=depth[6], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire8 = self._bsconv_layer('fire8', fire7, oc=depth[7], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire9 = self._bsconv_layer('fire9', fire8, oc=depth[8], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                   pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire10 = self._bsconv_layer('fire10', fire9, oc=depth[9], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire11 = self._bsconv_layer('fire11', fire10, oc=depth[10], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire12 = self._bsconv_layer('fire12', fire11, oc=depth[11], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire13 = self._bsconv_layer('fire13', fire12, oc=depth[12], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire14 = self._bsconv_layer('fire14', fire13, oc=depth[13], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire15 = self._bsconv_layer('fire15', fire14, oc=depth[14], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=True, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire16 = self._bsconv_layer('fire16', fire15, oc=depth[15], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire17 = self._bsconv_layer('fire17', fire16, oc=depth[16], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire18 = self._bsconv_layer('fire18', fire17, oc=depth[17], freeze=False, w_bin=ml_w_bin, a_bin=ml_a_bin,
                                    pool_en=False, min_rng=min_rng, max_rng=max_rng, bias_on=bias_on, mul_f=mul_f)
        fire_o = fire18

        ####################################################################

        num_output = mc.ANCHOR_PER_GRID * (mc.CLASSES + 1 + 4)
        self.preds, _ = self._conv_layer('fire_o', fire_o, filters=num_output, size=3, stride=1,
                                         padding='SAME', xavier=False, relu=False, stddev=0.0001, w_bin=ll_w_bin,
                                         bias_on=bias_on, mul_f=mul_f)
        print('self.preds:', self.preds)

    def _fire_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True,
                    min_rng=-0.5, max_rng=0.5, bias_on=False, mul_f=1):
        with tf.variable_scope(layer_name):
            ex3x3, _ = self._conv_layer('conv3x3', inputs, filters=oc, size=3, stride=1,
                                        padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin,
                                        bias_on=bias_on, mul_f=mul_f)

            ex3x3 = self._batch_norm('bn', ex3x3)  # <----
            ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)  # <---- relu
            tf.summary.histogram('after_relu', ex3x3)
            if pool_en:
                pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
            else:
                pool = ex3x3
            tf.summary.histogram('pool', pool)

            return pool

    def _mobile_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True,
                      min_rng=-0.5, max_rng=0.5, bias_on=False, mul_f=1):
        with tf.variable_scope(layer_name):
            ex3x3, _ = self._conv_layer('dw_conv3x3', inputs, filters=int(inputs.get_shape()[3]), size=3, stride=1,
                                        padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin,
                                        depthwise=True, bias_on=bias_on, mul_f=mul_f)  # <----

            tf.summary.histogram('dw_before_bn', ex3x3)
            ex3x3 = self._batch_norm('dw_bn', ex3x3)  # <----
            tf.summary.histogram('dw_before_relu', ex3x3)
            ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)  # <---- relu
            tf.summary.histogram('dw_after_relu', ex3x3)

            if pool_en:
                pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
            else:
                pool = ex3x3
            tf.summary.histogram('dw_pool', pool)

            ex1x1, _ = self._conv_layer('conv1x1', pool, filters=oc, size=1, stride=1,
                                        padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin,
                                        depthwise=False, bias_on=bias_on, mul_f=mul_f)  # <----

            tf.summary.histogram('1x1_before_bn', ex1x1)
            ex1x1 = self._batch_norm('1x1_bn', ex1x1)  # <----
            tf.summary.histogram('1x1_before_relu', ex1x1)
            ex1x1 = self.binary_wrapper(ex1x1, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)  # <---- relu
            tf.summary.histogram('1x1_after_relu', ex1x1)

            return ex1x1

    def _bsconv_layer(self, layer_name, inputs, oc, stddev=0.01, freeze=False, w_bin=16, a_bin=16, pool_en=True,
                      min_rng=-0.5, max_rng=0.5, bias_on=False, mul_f=1):
        with tf.variable_scope(layer_name):
            ex1x1, _ = self._conv_layer('conv1x1', inputs, filters=oc, size=1, stride=1,
                                        padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin,
                                        depthwise=False, bias_on=bias_on, mul_f=mul_f)

            # tf.summary.histogram('1x1_before_bn', ex1x1)
            ex1x1 = self._batch_norm('1x1_bn', ex1x1)  # <----
            # tf.summary.histogram('1x1_before_relu', ex1x1)
            ex1x1 = self.binary_wrapper(ex1x1, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)  # <---- relu
            # tf.summary.histogram('1x1_after_relu', ex1x1)

            ex3x3, _ = self._conv_layer('dw_conv3x3', ex1x1, filters=oc, size=3, stride=1,
                                        padding='SAME', stddev=stddev, freeze=freeze, relu=False, w_bin=w_bin,
                                        depthwise=True, bias_on=bias_on, mul_f=mul_f)  # <----

            # tf.summary.histogram('dw_before_bn', ex3x3)
            ex3x3 = self._batch_norm('dw_bn', ex3x3)  # <----
            # tf.summary.histogram('dw_before_relu', ex3x3)
            ex3x3 = self.binary_wrapper(ex3x3, a_bin=a_bin, min_rng=min_rng, max_rng=max_rng)  # <---- relu
            # tf.summary.histogram('dw_after_relu', ex3x3)

            if pool_en:
                pool = self._pooling_layer('pool', ex3x3, size=2, stride=2, padding='SAME')
            else:
                pool = ex3x3
            tf.summary.histogram('dw_pool', pool)

            return pool
