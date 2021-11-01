# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""SqueezeDet Demo.

In image detection mode, for a given image, detect objects and draw bounding
boxes around them. In video detection mode, perform real-time detection on the
video stream.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import glob
import numpy as np
import os
import skvideo.io
import sys
import tensorflow as tf
import time

from config import *
# from train import _draw_box
from nets import *
from utils.util import sparse_to_dense, bgr_to_rgb, bbox_transform

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'mode', 'image', """'image' or 'video'.""")
tf.app.flags.DEFINE_string(
    'checkpoint', './data/model_checkpoints/squeezeDet/model.ckpt-87000',
    """Path to the model parameter file.""")
tf.app.flags.DEFINE_string(
    'input_path', './data/sample.png',
    """Input image or video to be detected. Can process glob input such as """
    """./data/00000*.png.""")
tf.app.flags.DEFINE_string(
    'out_dir', './data/out/', """Directory to dump output image or video.""")
tf.app.flags.DEFINE_string(
    'demo_net', 'squeezeDet', """Neural net architecture.""")


# tf.app.flags.DEFINE_integer(
#    'gpu', 1, """GPU selection.""")

def _draw_box(im, box_list, label_list, color=(128, 0, 128), cdict=None, form='center', scale=1):
    assert form == 'center' or form == 'diagonal', \
        'bounding box format not accepted: {}.'.format(form)

    for bbox, label in zip(box_list, label_list):

        if form == 'center':
            bbox = bbox_transform(bbox)

        xmin, ymin, xmax, ymax = [int(b) * scale for b in bbox]

        l = label.split(':')[0]  # text before "CLASS: (PROB)"
        if cdict and l in cdict:
            c = cdict[l]
        else:
            c = color

        # draw box
        cv2.rectangle(im, (xmin, ymin), (xmax, ymax), c, 2 * scale)
        # draw label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(im, label, (max(1, xmin - 10), ymax + 10), font, 0.5 * scale, c, 2 * scale)  # <--------------------


def video_demo():
    """Detect videos."""

    cap = cv2.VideoCapture(FLAGS.input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  # float

    out_name = FLAGS.input_path.split('/')[-1:][0]
    out_name = out_name.split('.')[0]
    out_name = os.path.join(FLAGS.out_dir, 'det_' + out_name + '.avi')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video = cv2.VideoWriter(out_name, fourcc, fps, (int(width), int(height)), True)

    assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
        'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

    with tf.Graph().as_default():
        # Load model
        if FLAGS.demo_net == 'squeezeDet':
            mc = kitti_squeezeDet_config()
            mc.BATCH_SIZE = 1
            mc.IS_TRAINING = False
            # model parameters will be restored from checkpoint
            mc.LOAD_PRETRAINED_MODEL = False
            mc.PLOT_PROB_THRESH = 0.2
            # model = SqueezeDet(mc, FLAGS.gpu)
            model = SqueezeDet(mc, gpu_id=0)
        elif FLAGS.demo_net == 'squeezeDet+':
            mc = kitti_squeezeDetPlus_config()
            mc.BATCH_SIZE = 1
            mc.LOAD_PRETRAINED_MODEL = False
            mc.IS_TRAINING = False
            model = SqueezeDetPlus(mc, FLAGS.gpu)

        saver = tf.train.Saver(model.model_params)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)

            # ==============================================================================
            # Store Graph
            # ==============================================================================
            tf.train.write_graph(sess.graph_def, "/tmp/tensorflow", "test.pb", as_text=False)
            tf.train.write_graph(sess.graph_def, "/tmp/tensorflow", "test.pbtxt", as_text=True)
            # ==============================================================================
            print("Graph store is done")

            times = {}
            count = 0
            det_last = [0, 0, 0]
            top_last = [0, 0, 0]
            while cap.isOpened():
                t_start = time.time()
                count += 1

                # Load images from video and crop
                ret, frame = cap.read()  # <--- RGB format
                if ret == True:
                    frame = frame[:, :, ::-1]  # <--- convert to BGR
                    orig_h, orig_w, _ = [float(v) for v in frame.shape]
                    scale_h = int(orig_h / mc.IMAGE_HEIGHT)
                    scale_w = int(orig_w / mc.IMAGE_WIDTH)
                    up_scale = min(scale_h, scale_w)

                    y_start = int(orig_h / 2 - mc.IMAGE_HEIGHT * up_scale / 2)
                    x_start = int(orig_w / 2 - mc.IMAGE_WIDTH * up_scale / 2)
                    im = frame[y_start:y_start + mc.IMAGE_HEIGHT * up_scale,
                         x_start:x_start + mc.IMAGE_WIDTH * up_scale]
                    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                    im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                    im = im.reshape((mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1))
                    im_input = im / 128.0

                else:
                    print('Done')
                    break

                t_reshape = time.time()
                times['reshape'] = t_reshape - t_start

                # Detect
                det_boxes, det_probs, det_class = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class],
                    feed_dict={model.image_input: [im_input]})

                t_detect = time.time()
                times['detect'] = t_detect - t_reshape

                # Extract class only - mine :)
                top_idx = det_probs[0].argsort()[:-2:-1]  # top probability only
                top_prob = det_probs[0][top_idx]
                top_class = det_class[0][top_idx]
                if (top_prob > mc.PLOT_PROB_THRESH):
                    new_top_last = [top_last[1], top_last[2], 1]
                else:
                    new_top_last = [top_last[1], top_last[2], 0]
                # End of mine

                # Filter
                final_boxes, final_probs, final_class = model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

                keep_idx = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]

                frame = frame[:, :, ::-1]
                im_show = frame[y_start:y_start + mc.IMAGE_HEIGHT * up_scale,
                          x_start:x_start + mc.IMAGE_WIDTH * up_scale]

                if (len(keep_idx) != 0):
                    final_boxes = [final_boxes[idx] for idx in keep_idx]
                    final_probs = [final_probs[idx] for idx in keep_idx]
                    final_class = [final_class[idx] for idx in keep_idx]

                    t_filter = time.time()
                    times['filter'] = t_filter - t_detect

                    # Draw boxes
                    # TODO(bichen): move this color dict to configuration file
                    cls2clr = {
                        'front_user': (128, 255, 0),
                        'other_user': (0, 128, 155),
                        'front_surf': (255, 128, 0),
                        'other_surf': (128, 0, 255)
                    }
                    print(final_boxes)

                    if (sum(det_last) != 0):  # filter
                        _draw_box(
                            im_show, final_boxes,
                            [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
                             for idx, prob in zip(final_class, final_probs)], color=(255, 255, 0), scale=up_scale,
                            cdict=cls2clr
                        )

                im_show_exp = im_show
                frame[y_start:y_start + mc.IMAGE_HEIGHT * up_scale,
                x_start:x_start + mc.IMAGE_WIDTH * up_scale] = im_show_exp
                cv2.rectangle(frame, (x_start, y_start),
                              (x_start + mc.IMAGE_WIDTH * up_scale, y_start + mc.IMAGE_HEIGHT * up_scale),
                              (255, 0, 255), 2)

                if (top_prob > mc.PLOT_PROB_THRESH and sum(top_last) != 0):
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    print('top_class=', top_class[0])
                    label = mc.CLASS_NAMES[top_class[0]]  # +': (%.2f)'% top_prob[0]
                    label = label[-2:]
                    cv2.putText(frame, label, (x_start, y_start), font, 1.5, (0, 255, 0), 2)

                cv2.imshow('video', frame)  # <--- RGB input
                video.write(frame)
                if (len(keep_idx) != 0 and sum(det_last) != 0):
                    for x in range(10):  # slow down in demo video
                        video.write(frame)

                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                new_det_last = [det_last[1], det_last[2], len(keep_idx)]
                det_last = new_det_last
                top_last = new_top_last
    # Release everything if job is finished
    cap.release()
    video.release()
    cv2.destroyAllWindows()


def image_demo():
    """Detect image."""

    assert FLAGS.demo_net == 'squeezeDet' or FLAGS.demo_net == 'squeezeDet+', \
        'Selected nueral net architecture not supported: {}'.format(FLAGS.demo_net)

    with tf.Graph().as_default():
        # Load model
        if FLAGS.demo_net == 'squeezeDet':
            mc = kitti_squeezeDet_config()
            mc.BATCH_SIZE = 1
            # model parameters will be restored from checkpoint
            mc.LOAD_PRETRAINED_MODEL = False
            mc.IS_TRAINING = False
            mc.PLOT_PROB_THRESH = 0.1
            # model = SqueezeDet(mc, FLAGS.gpu)
            model = SqueezeDet(mc, gpu_id=1)
        elif FLAGS.demo_net == 'squeezeDet+':
            mc = kitti_squeezeDetPlus_config()
            mc.BATCH_SIZE = 1
            mc.LOAD_PRETRAINED_MODEL = False
            mc.IS_TRAINING = False
            model = SqueezeDetPlus(mc, FLAGS.gpu)

        saver = tf.train.Saver(model.model_params)

        np.set_printoptions(threshold=sys.maxsize)

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            saver.restore(sess, FLAGS.checkpoint)

            # ==============================================================================
            # Store Graph
            # ==============================================================================
            tf.train.write_graph(sess.graph_def, "/tmp/tensorflow", "test.pb", as_text=False)
            tf.train.write_graph(sess.graph_def, "/tmp/tensorflow", "test.pbtxt", as_text=True)
            # ==============================================================================

            for f in glob.iglob(FLAGS.input_path):
                print('file name:' + f)
                im = cv2.imread(f)  # <---------------------------- BGR format
                im = im.astype(np.float32, copy=False)

                im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
                org_im = im.copy()
                org_im = cv2.cvtColor(org_im, cv2.COLOR_GRAY2BGR)
                im = im.reshape((mc.IMAGE_HEIGHT, mc.IMAGE_WIDTH, 1))
                orig_h, orig_w, _ = [float(v) for v in im.shape]
                # im -= mc.BGR_MEANS # <---------------------------------------------------------------------!!!!!!
                int_im = im.copy()
                int_im = int_im.astype(np.uint8)
                int_im = np.reshape(int_im, -1)
                hex_im = '\n'.join(format(x, '02x') for x in int_im)
                im /= 128.0

                # Detect
                '''
                det_boxes, det_probs, det_class = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class],
                    feed_dict={model.image_input:[im]})
                '''
                det_boxes, det_probs, det_class, conf, class_probs, preds = sess.run(
                    [model.det_boxes, model.det_probs, model.det_class, model.pred_conf, model.pred_class_probs,
                     model.preds],
                    feed_dict={model.image_input: [im]})

                print("conf: ", conf.shape)
                print(conf)
                print("class_probs: ", class_probs.shape)
                print(class_probs)
                print("preds: ", preds.shape)
                print(preds)

                # Filter
                final_boxes, final_probs, final_class = model.filter_prediction(
                    det_boxes[0], det_probs[0], det_class[0])

                keep_idx = [idx for idx in range(len(final_probs)) \
                            if final_probs[idx] > mc.PLOT_PROB_THRESH]
                final_boxes = [final_boxes[idx] for idx in keep_idx]
                final_probs = [final_probs[idx] for idx in keep_idx]
                final_class = [final_class[idx] for idx in keep_idx]

                # TODO(bichen): move this color dict to configuration file
                cls2clr = {
                    'front_user': (128, 255, 0),
                    'other_user': (0, 128, 255),
                    'front_surf': (255, 128, 0),
                    'other_surf': (128, 0, 255)
                }

                # Draw boxes
                print('# of final boxes=', len(keep_idx))
                _draw_box(
                    # im_gray, final_boxes,
                    org_im, final_boxes,
                    [mc.CLASS_NAMES[idx] + ': (%.2f)' % prob \
                     for idx, prob in zip(final_class, final_probs)],
                    (255, 255, 0), cdict=cls2clr
                )

                file_name = os.path.split(f)[1]
                print(file_name)
                out_file_name = os.path.join(FLAGS.out_dir, 'out_' + file_name)

                cv2.imwrite(out_file_name, org_im)  # <----- BGR format
                print('Image detection output saved to {}'.format(out_file_name))


def main(argv=None):
    if not tf.gfile.Exists(FLAGS.out_dir):
        tf.gfile.MakeDirs(FLAGS.out_dir)
    if FLAGS.mode == 'image':
        image_demo()
    else:
        video_demo()


if __name__ == '__main__':
    tf.app.run()
