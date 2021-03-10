#!/usr/bin/env python3
#-------------------------------------------------------------------------------
# Author: Lukasz Janyst <lukasz@jany.st>
# Date:   07.09.2017
#-------------------------------------------------------------------------------
# This file is part of SSD-TensorFlow.
#
# SSD-TensorFlow is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# SSD-TensorFlow is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with SSD-Tensorflow.  If not, see <http://www.gnu.org/licenses/>.
#-------------------------------------------------------------------------------

import argparse
import math
import sys
import os

import multiprocessing as mp
import tensorflow as tf
import numpy as np

from average_precision import APCalculator, APs2mAP
from training_data import TrainingData
from ssdutils import get_anchors_for_preset, decode_boxes, suppress_overlaps
from ssdmv2 import SSDMV2
from utils import *
from tqdm import tqdm
import config
if sys.version_info[0] < 3:
    print("This is a Python 3 program. Use Python 3 or higher.")
    sys.exit(1)

#-------------------------------------------------------------------------------
def compute_lr(lr_values, lr_boundaries):
    with tf.variable_scope('learning_rate'):
        global_step = tf.Variable(0, trainable=False, name='global_step')
        lr = tf.train.piecewise_constant(global_step, lr_boundaries, lr_values)
    return lr, global_step

#-------------------------------------------------------------------------------
def main():
    #---------------------------------------------------------------------------
    # Parse the commandline
    #---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Train the SSD')
    parser.add_argument('--name', default='checkpoints',
                        help='checkpoints name')
    parser.add_argument('--data-dir', default='pascal-voc',
                        help='data directory')
    parser.add_argument('--epochs', type=int, default=1000,
                        help='number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='batch size')
    parser.add_argument('--tensorboard-dir', default="tb",
                        help='name of the tensorboard data directory')
    parser.add_argument('--checkpoint-interval', type=int, default=1,
                        help='checkpoint interval')
    parser.add_argument('--lr-values', type=str, default='0.001;0.0001;0.00001',
                        help='learning rate values')
    parser.add_argument('--lr-boundaries', type=str, default='320000;400000',
                        help='learning rate chage boundaries (in batches)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum for the optimizer')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='L2 normalization factor')
    # parser.add_argument('--continue-training', type=str2bool, default='False',
    #                     help='continue training from the latest checkpoint -  True or False')
    parser.add_argument('--num-workers', type=int, default=mp.cpu_count(),
                        help='number of parallel generators')
    # parser.add_argument('--generate-pbtxt', default=False, action="store_true",
    #                     help='Generate .pbtxt')

    args = parser.parse_args()

    print('[i] Checkpoints name:     ', args.name)
    print('[i] Data directory:       ', args.data_dir)
    print('[i] # epochs:             ', args.epochs)
    print('[i] Batch size:           ', args.batch_size)
    print('[i] Tensorboard directory:', args.name)
    print('[i] Checkpoint interval:  ', args.checkpoint_interval)
    print('[i] Learning rate values: ', args.lr_values)
    print('[i] Learning rate boundaries: ', args.lr_boundaries)
    print('[i] Momentum:             ', args.momentum)
    print('[i] Weight decay:         ', args.weight_decay)
    # print('[i] Continue:             ', args.continue_training)
    print('[i] Number of workers:    ', args.num_workers)

    #---------------------------------------------------------------------------
    # Find an existing checkpoint
    #---------------------------------------------------------------------------
    start_epoch = 0
    state_flag = 1
    ckpt_flag = 1
    ckpt_file_flag = 1
    meta_file_flag =1


    if os.path.exists(args.name):
        state = tf.train.get_checkpoint_state(args.name)
        if state is None:
            print('[!] No network state found in ' + args.name)
            state_flag = 0
            return 1

        ckpt_paths = state.all_model_checkpoint_paths
        if not ckpt_paths:

            print('[!] No network state found in ' + args.name)
            ckpt_flag = 0
            return 1

        last_epoch = None
        checkpoint_file = None
        for ckpt in ckpt_paths:
            ckpt_num = os.path.basename(ckpt).split('.')[0][1:]
            try:
                ckpt_num = int(ckpt_num)
            except ValueError:
                continue
            if last_epoch is None or last_epoch < ckpt_num:
                last_epoch = ckpt_num
                checkpoint_file = ckpt

        if checkpoint_file is None:
            print('[!] No checkpoints found, cannot continue!')
            ckpt_file_flag = 0
            return 1

        metagraph_file = checkpoint_file + '.meta'

        if not os.path.exists(metagraph_file):
            print('[!] Cannot find metagraph', metagraph_file)
            meta_file_flag = 0
            return 1
        #start_epoch = last_epoch



        if ckpt_flag and state_flag and ckpt_file_flag and meta_file_flag :
        	start_epoch = last_epoch







        

    # elif args.generate_pbtxt:
    #     print("Generating pbtxt!")
    #---------------------------------------------------------------------------
    # Create a project directory
    #---------------------------------------------------------------------------
    else:
        try:
            print('[i] Creating directory {}...'.format(args.name))
            if not os.path.exists(args.name):
                os.makedirs(args.name)
        except (IOError) as e:
            print('[!]', str(e))
            return 1

    print('[i] Starting at epoch:    ', start_epoch+1)

    #---------------------------------------------------------------------------
    # Configure the training data
    #---------------------------------------------------------------------------
    print('[i] Configuring the training data...')
    try:
        td = TrainingData(args.data_dir)
        print('[i] # training samples:   ', td.num_train)
        print('[i] # validation samples: ', td.num_valid)
        print('[i] # classes:            ', td.num_classes)
        print('[i] Image size:           ', td.preset.image_size)
    except (AttributeError, RuntimeError) as e:
        print('[!] Unable to load training data:', str(e))
        return 1

    #---------------------------------------------------------------------------
    # Create the network
    #---------------------------------------------------------------------------
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session() as sess:
    #with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print('[i] Creating the model...')
        n_train_batches = int(math.ceil(td.num_train/args.batch_size))
        n_valid_batches = int(math.ceil(td.num_valid/args.batch_size))

        global_step = None
        if start_epoch == 0:
            lr_values = args.lr_values.split(';')
            try:
                lr_values = [float(x) for x in lr_values]
            except ValueError:
                print('[!] Learning rate values must be floats')
                sys.exit(1)

            lr_boundaries = args.lr_boundaries.split(';')
            try:
                lr_boundaries = [int(x) for x in lr_boundaries]
            except ValueError:
                print('[!] Learning rate boundaries must be ints')
                sys.exit(1)

            ret = compute_lr(lr_values, lr_boundaries)
            learning_rate, global_step = ret

        is_training = True

        net = SSDMV2(sess, td.preset, is_training)
        if start_epoch != 0:
            net.build_from_metagraph(metagraph_file, checkpoint_file)
            net.build_optimizer_from_metagraph()
        else:
            net.build_from_mv2(td.num_classes)
            net.build_optimizer(learning_rate=learning_rate,
                                global_step=global_step,
                                weight_decay=args.weight_decay,
                                momentum=args.momentum)
        # if args.generate_pbtxt:
        #     tf.train.write_graph(sess.graph_def, "./", 'inference_graph.pbtxt', as_text=True)
        #     sys.exit()

        initialize_uninitialized_variables(sess)

        #-----------------------------------------------------------------------
        # Create various helpers
        #-----------------------------------------------------------------------
        summary_writer = tf.summary.FileWriter(args.name,
                                               sess.graph)
        saver = tf.train.Saver(max_to_keep=20)

        anchors = get_anchors_for_preset(td.preset)

        training_ap_calc = APCalculator()
        validation_ap_calc = APCalculator()

        #-----------------------------------------------------------------------
        # Summaries
        #-----------------------------------------------------------------------
        restore = start_epoch != 0

        training_ap = PrecisionSummary(sess, summary_writer, 'training',
                                       td.lname2id.keys(), restore)
        validation_ap = PrecisionSummary(sess, summary_writer, 'validation',
                                         td.lname2id.keys(), restore)

        training_imgs = ImageSummary(sess, summary_writer, 'training',
                                     td.label_colors, restore)
        validation_imgs = ImageSummary(sess, summary_writer, 'validation',
                                       td.label_colors, restore)

        training_loss = LossSummary(sess, summary_writer, 'training',
                                    td.num_train, restore)
        validation_loss = LossSummary(sess, summary_writer, 'validation',
                                      td.num_valid, restore)

        #-----------------------------------------------------------------------
        # Get the initial snapshot of the network
        #-----------------------------------------------------------------------
        net_summary_ops = net.build_summaries(restore)
        if start_epoch == 0:
            net_summary = sess.run(net_summary_ops)
            summary_writer.add_summary(net_summary, 0)
        summary_writer.flush()

        #-----------------------------------------------------------------------
        # Cycle through the epoch
        #-----------------------------------------------------------------------
        print('[i] Training...')
        for e in range(start_epoch, args.epochs):
            training_imgs_samples = []
            validation_imgs_samples = []

            #-------------------------------------------------------------------
            # Train
            #-------------------------------------------------------------------
            generator = td.train_generator(args.batch_size, args.num_workers)
            description = '[i] Train {:>2}/{}'.format(e+1, args.epochs)
            for x, y, gt_boxes in tqdm(generator, total=n_train_batches,
                                       desc=description, unit='batches'):

                if len(training_imgs_samples) < 3:
                    saved_images = np.copy(x[:3])
                feed = {net.image_input: x,
                        net.labels: y}
                result, training_loss_batch, _ = sess.run([net.result, net.losses,
                                                  net.optimizer],
                                                 feed_dict=feed)

                if math.isnan(training_loss_batch['confidence']):
                    print('[!] Confidence loss is NaN.')

                training_loss.add(training_loss_batch, x.shape[0])

                if e == 0: continue

                for i in range(result.shape[0]):
                    boxes = decode_boxes(result[i], anchors, 0.5, td.lid2name)
                    boxes = suppress_overlaps(boxes)
                    training_ap_calc.add_detections(gt_boxes[i], boxes)

                    if len(training_imgs_samples) < 3:
                        training_imgs_samples.append((saved_images[i] * config.image_scale, boxes))

            #-------------------------------------------------------------------
            # Validate
            #-------------------------------------------------------------------
            generator = td.valid_generator(args.batch_size, args.num_workers)
            description = '[i] Valid {:>2}/{}'.format(e+1, args.epochs)

            for x, y, gt_boxes in tqdm(generator, total=n_valid_batches,
                                       desc=description, unit='batches'):
                feed = {net.image_input: x,
                        net.labels: y}
                result, validation_loss_batch = sess.run([net.result, net.losses],
                                              feed_dict=feed)

                validation_loss.add(validation_loss_batch,  x.shape[0])

                if e == 0: continue

                for i in range(result.shape[0]):
                    boxes = decode_boxes(result[i], anchors, 0.5, td.lid2name)
                    boxes = suppress_overlaps(boxes)
                    validation_ap_calc.add_detections(gt_boxes[i], boxes)

                    if len(validation_imgs_samples) < 3:
                        validation_imgs_samples.append(((np.copy(x[i])) * config.image_scale, boxes))

            #-------------------------------------------------------------------
            # Write summaries
            #-------------------------------------------------------------------
            training_loss.push(e+1)
            validation_loss.push(e+1)

            net_summary = sess.run(net_summary_ops)
            summary_writer.add_summary(net_summary, e+1)

            APs = training_ap_calc.compute_aps()
            mAP = APs2mAP(APs)
            training_ap.push(e+1, mAP, APs)

            APs = validation_ap_calc.compute_aps()
            mAP = APs2mAP(APs)
            validation_ap.push(e+1, mAP, APs)

            training_ap_calc.clear()
            validation_ap_calc.clear()

            training_imgs.push(e+1, training_imgs_samples)
            validation_imgs.push(e+1, validation_imgs_samples)

            summary_writer.flush()

            if (e+1) % args.checkpoint_interval == 0:
                print("training loss:")
                print(training_loss_batch)
                print("validation loss:")
                print(validation_loss_batch)

            #-------------------------------------------------------------------
            # Save a checktpoint
            #-------------------------------------------------------------------
            if (e+1) % args.checkpoint_interval == 0:
                checkpoint = '{}/e{}.ckpt'.format(args.name, e+1)
                saver.save(sess, checkpoint)
                print('[i] Checkpoint saved:', checkpoint)

                name = "e"+str(e+1)+".pbtxt"
                tf.train.write_graph(sess.graph_def, args.name, name, as_text=True)


        checkpoint = '{}/e{}.ckpt'.format(args.name,int(args.epochs)-1)
        saver.save(sess, checkpoint)
        print('[i] Checkpoint saved:', checkpoint)
        name = "e"+str(e+1)+".txt"
        tf.train.write_graph(sess.graph_def, args.name, name, as_text=True)


    return 0

if __name__ == '__main__':
    sys.exit(main())
