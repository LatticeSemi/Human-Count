# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Image data base class for kitti"""

import cv2
import os 
import numpy as np
import subprocess

from dataset.imdb import imdb
from utils.util import bbox_transform_inv, batch_iou
from google.cloud import storage
from glob import glob
import tqdm
class kitti_gcp(imdb):
  def __init__(self, image_set,bucket_name,bucket_data_path,dataset_path, mc):
    imdb.__init__(self, 'kitti_'+image_set, mc)
    self._image_set = image_set
    self._data_root_path = [bucket_data_path]
    self._image_path = [os.path.join(path, 'training', 'images') for path in self._data_root_path]
    self._label_path = [os.path.join(path, 'training', 'labels') for path in self._data_root_path]
    self._classes = self.mc.CLASS_NAMES
    print(dataset_path)
    json_path = glob(dataset_path+"/**.json")[0]
    self._class_to_idx = dict(zip(self.classes, range(self.num_classes)))
    self.initialize_gcp(json_path,bucket_name)
    # a list of string indices of images in the directory
    self._image_idx,self._label_idx = self._load_image_set_idx() 
    
    # a dict of image_idx -> [[cx, cy, w, h, cls_idx]]. x,y,w,h are not divided by
    # the image width and height
    self._rois = self._load_kitti_annotation()

    ## batch reader ##
    self._perm_idx = None
    self._cur_idx = 0
    # TODO(bichen): add a random seed as parameter
    self._shuffle_image_idx()

    self._eval_tool = './src/dataset/kitti-eval/cpp/evaluate_object'
    
  def initialize_gcp(self,path_to_json,bucket_name):
    
    self.storage_client = storage.Client.from_service_account_json(path_to_json)
    self.bucket = self.storage_client.get_bucket(bucket_name)
    # get bucket data as blob
    
  def download_txt(self,filename):
    blob = self.bucket.blob(filename)
    downloaded_blob = blob.download_as_string()
    downloaded_blob = downloaded_blob.decode("utf-8") 
    return downloaded_blob
  
  def download_image(self,filename):
    blob = self.bucket.blob(filename)
    downloaded_blob = blob.download_as_bytes()
    img = np.fromstring(downloaded_blob, dtype='uint8')
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img
    
  def read_batch(self, shuffle=True):
        """Read a batch of image and bounding box annotations.
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      image_per_batch: images. Shape: batch_size x width x height x [b, g, r]
      label_per_batch: labels. Shape: batch_size x object_num
      delta_per_batch: bounding box deltas. Shape: batch_size x object_num x
          [dx ,dy, dw, dh]
      aidx_per_batch: index of anchors that are responsible for prediction.
          Shape: batch_size x object_num
      bbox_per_batch: scaled bounding boxes. Shape: batch_size x object_num x
          [cx, cy, w, h]
    """
        mc = self.mc

        if shuffle:
            if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
                self._shuffle_image_idx()
            batch_idx = self._perm_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
            self._cur_idx += mc.BATCH_SIZE
        else:
            if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
                batch_idx = self._image_idx[self._cur_idx:] \
                            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE - len(self._image_idx)]
                self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
            else:
                batch_idx = self._image_idx[self._cur_idx:self._cur_idx + mc.BATCH_SIZE]
                self._cur_idx += mc.BATCH_SIZE

        image_per_batch = []
        image_per_batch_viz = []
        label_per_batch = []
        bbox_per_batch = []
        delta_per_batch = []
        aidx_per_batch = []
        if mc.DEBUG_MODE:
            avg_ious = 0.
            num_objects = 0.
            max_iou = 0.0
            min_iou = 1.0
            num_zero_iou_obj = 0

        for idx in batch_idx:
            # load the image
            im = self.download_image(self._image_path_at(idx))
            if im is None:
                print('failed file read:' + self._image_path_at(idx))

            im = im.astype(np.float32, copy=False)

            # random brightness control
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            add_v = np.random.randint(55, 200) - 128
            v = np.where(v <= 255 - add_v, v + add_v, 255)
            final_hsv = cv2.merge((h, s, v))
            im = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

            im -= mc.BGR_MEANS  # <-------------------------------
            im /= 128.0  # to make input in the range of [0, 2)
            orig_h, orig_w, _ = [float(v) for v in im.shape]
            if mc.GRAY:
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

            # load annotations
            label_per_batch.append([b[4] for b in self._rois[idx][:]])
            gt_bbox = np.array(
                [[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[2] - b[0], b[3] - b[1]] for b in self._rois[idx][:]])

            assert np.all(gt_bbox[:, 0]) > 0, 'less than 0 gt_bbox[0]'
            assert np.all(gt_bbox[:, 1]) > 0, 'less than 0 gt_bbox[1]'
            assert np.all(gt_bbox[:, 2]) > 0, 'less than 0 gt_bbox[2]'
            assert np.all(gt_bbox[:, 3]) > 0, 'less than 0 gt_bbox[3]'

            if mc.DATA_AUGMENTATION:
                # Flip image with 50% probability
                if np.random.randint(2) > 0.5 and not mc.GRAY:
                    im = im[:, ::-1, :]
                else:
                    im = im[:, ::-1]
                gt_bbox[:, 0] = orig_w - 1 - gt_bbox[:, 0]

            # scale image
            im = cv2.resize(im, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT), interpolation=cv2.INTER_AREA)
            if mc.GRAY:
                im = im.reshape((mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT, 1))
            image_per_batch.append(im)
            image_per_batch_viz.append(im * 128.0)

            # scale annotation
            x_scale = mc.IMAGE_WIDTH / orig_w
            y_scale = mc.IMAGE_HEIGHT / orig_h
            gt_bbox[:, 0::2] = gt_bbox[:, 0::2] * x_scale
            gt_bbox[:, 1::2] = gt_bbox[:, 1::2] * y_scale
            bbox_per_batch.append(gt_bbox)

            aidx_per_image, delta_per_image = [], []
            aidx_set = set()
            for i in range(len(gt_bbox)):
                overlaps = batch_iou(mc.ANCHOR_BOX, gt_bbox[i])

                aidx = len(mc.ANCHOR_BOX)
                for ov_idx in np.argsort(overlaps)[::-1]:
                    if overlaps[ov_idx] <= 0:
                        if mc.DEBUG_MODE:
                            min_iou = min(overlaps[ov_idx], min_iou)
                            num_objects += 1
                            num_zero_iou_obj += 1
                        break
                    if ov_idx not in aidx_set:
                        aidx_set.add(ov_idx)
                        aidx = ov_idx
                        if mc.DEBUG_MODE:
                            max_iou = max(overlaps[ov_idx], max_iou)
                            min_iou = min(overlaps[ov_idx], min_iou)
                            avg_ious += overlaps[ov_idx]
                            num_objects += 1
                        break

                if aidx == len(mc.ANCHOR_BOX):
                    # even the largeset available overlap is 0, thus, choose one with the
                    # smallest square distance
                    dist = np.sum(np.square(gt_bbox[i] - mc.ANCHOR_BOX), axis=1)
                    for dist_idx in np.argsort(dist):
                        if dist_idx not in aidx_set:
                            aidx_set.add(dist_idx)
                            aidx = dist_idx
                            break

                box_cx, box_cy, box_w, box_h = gt_bbox[i]
                delta = [0] * 4
                delta[0] = (box_cx - mc.ANCHOR_BOX[aidx][0]) / mc.ANCHOR_BOX[aidx][2]
                delta[1] = (box_cy - mc.ANCHOR_BOX[aidx][1]) / mc.ANCHOR_BOX[aidx][3]
                if False:
                    delta[2] = np.log(box_w / mc.ANCHOR_BOX[aidx][2])
                    delta[3] = np.log(box_h / mc.ANCHOR_BOX[aidx][3])
                else:  # to remove exp in FPGA
                    delta[2] = box_w / mc.ANCHOR_BOX[aidx][2]
                    delta[3] = box_h / mc.ANCHOR_BOX[aidx][3]

                aidx_per_image.append(aidx)
                delta_per_image.append(delta)

            delta_per_batch.append(delta_per_image)
            aidx_per_batch.append(aidx_per_image)

        if mc.DEBUG_MODE:
            print('max iou: {}'.format(max_iou))
            print('min iou: {}'.format(min_iou))
            print('avg iou: {}'.format(avg_ious / num_objects))
            print('number of objects: {}'.format(num_objects))
            print('number of objects with 0 iou: {}'.format(num_zero_iou_obj))

        return image_per_batch, label_per_batch, delta_per_batch, \
               aidx_per_batch, bbox_per_batch, image_per_batch_viz

  def _load_image_set_idx(self):
    self._image_path_idx = {}
    image_set_file = [os.path.join(
        data_root_path, 'ImageSets', self._image_set+'.txt') for data_root_path in self._data_root_path]
    #for set_file in image_set_file:
    #    assert os.path.exists(set_file), \
    #        'File does not exist: {}'.format(set_file)
    image_idx = []
    label_idx = []
    idx = []
    for root_file,label_path,set_file in zip(self._image_path,self._label_path,image_set_file):
        text = self.download_txt(set_file)
        
        idx.extend(text.split()[:100]) 
        image_idx.extend([os.path.join(root_file,x+'.jpg') for x in idx])
        label_idx.extend([os.path.join(label_path,x+'.txt') for x in idx])
        
          
    for ids,image_id in zip(idx,image_idx):
        self._image_path_idx[ids] = image_id
    return idx,label_idx

  def _image_path_at(self, idx):
    image_path =self._image_path_idx[idx]
    #assert os.path.exists(image_path), \
    #    'Image does not exist: {}'.format(image_path)
    return image_path

  def _load_kitti_annotation(self):
    '''
    def _get_obj_level(obj):
      height = float(obj[7]) - float(obj[5]) + 1
      truncation = float(obj[1])
      occlusion = float(obj[2])
      if height >= 40 and truncation <= 0.15 and occlusion <= 0:
          return 1
      elif height >= 25 and truncation <= 0.3 and occlusion <= 1:
          return 2
      elif height >= 25 and truncation <= 0.5 and occlusion <= 2:
          return 3
      else:
          return 4
    '''

    idx2annotation = {}
    max_person = 0
    print("Loading labels from aws, this might take some time")
    for idx,index in tqdm.tqdm(zip(self._image_idx,self._label_idx)):
      filename = index
      #with open(filename, 'r') as f:
      #  lines = f.readlines()
      #f.close()
      lines = self.download_txt(filename)
      lines = lines.split('\n') 
      bboxes = []
      num_person = 0
      for line in lines:
        if line =='':
            continue
        obj = line.strip().split(' ')
        #try:
        cls = self._class_to_idx[obj[0].lower().strip()] # person (14) -> class 0
        #except Exception as e:
        #  print("Error",e)
        #  continue

        num_person = num_person + 1
        x, y, w, h = float(obj[4]), float(obj[5]), float(obj[6]), float(obj[7])
        bboxes.append([x, y, w, h, cls])

      assert len(bboxes) > 0, 'empty box image'
      if num_person > max_person:
        max_person = num_person
      idx2annotation[idx] = bboxes

    print('max person:', max_person)
    return idx2annotation

  def evaluate_detections(self, eval_dir, global_step, all_boxes):
    """Evaluate detection results.
    Args:
      eval_dir: directory to write evaluation logs
      global_step: step of the checkpoint
      all_boxes: all_boxes[cls][image] = N x 5 arrays of 
        [xmin, ymin, xmax, ymax, score]
    Returns:
      aps: array of average precisions.
      names: class names corresponding to each ap
    """
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step), 'data/')
    print('det_file_dir: '+det_file_dir)
    if not os.path.isdir(det_file_dir):
      print('make new directory')
      os.makedirs(det_file_dir)
      os.system('cp -r /data/trafficsignal/lisa/training/label_2/vid* '+det_file_dir)

    for im_idx, index in enumerate(self._image_idx):
      filename = os.path.join(det_file_dir, index+'.txt')
      with open(filename, 'wt') as f:
        for cls_idx, cls in enumerate(self._classes):
          dets = all_boxes[cls_idx][im_idx]
          for k in xrange(len(dets)):
            f.write(
                '{:s} -1 -1 0.0 {:.2f} {:.2f} {:.2f} {:.2f} 0.0 0.0 0.0 0.0 0.0 '
                '0.0 0.0 {:.3f}\n'.format(
                    cls.lower(), dets[k][0], dets[k][1], dets[k][2], dets[k][3],
                    dets[k][4])
            )

    cmd = self._eval_tool + ' ' \
          + os.path.join(self._data_root_path, 'training') + ' ' \
          + os.path.join(self._data_root_path, 'ImageSets',
                         self._image_set+'.txt') + ' ' \
          + os.path.dirname(det_file_dir) + ' ' + str(len(self._image_idx))

    print('Running: {}'.format(cmd))
    status = subprocess.call(cmd, shell=True)

    aps = []
    names = []
    for cls in self._classes:
      det_file_name = os.path.join(
          os.path.dirname(det_file_dir), 'stats_{:s}_ap.txt'.format(cls))
      if os.path.exists(det_file_name):
        with open(det_file_name, 'r') as f:
          lines = f.readlines()
        assert len(lines) == 3, \
            'Line number of {} should be 3'.format(det_file_name)

        aps.append(float(lines[0].split('=')[1].strip()))
        aps.append(float(lines[1].split('=')[1].strip()))
        aps.append(float(lines[2].split('=')[1].strip()))
      else:
        aps.extend([0.0, 0.0, 0.0])

      names.append(cls+'_easy')
      names.append(cls+'_medium')
      names.append(cls+'_hard')

    return aps, names

  def do_detection_analysis_in_eval(self, eval_dir, global_step):
    det_file_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step), 'data')
    det_error_dir = os.path.join(
        eval_dir, 'detection_files_{:s}'.format(global_step),
        'error_analysis')
    if not os.path.exists(det_error_dir):
      os.makedirs(det_error_dir)
    det_error_file = os.path.join(det_error_dir, 'det_error_file.txt')

    stats = self.analyze_detections(det_file_dir, det_error_file)
    ims = self.visualize_detections(
        image_dir=self._image_path,
        image_format='.jpg',
        det_error_file=det_error_file,
        output_image_dir=det_error_dir,
        num_det_per_type=10
    )

    return stats, ims

  def analyze_detections(self, detection_file_dir, det_error_file):
    def _save_detection(f, idx, error_type, det, score):
      f.write(
          '{:s} {:s} {:.1f} {:.1f} {:.1f} {:.1f} {:s} {:.3f}\n'.format(
              idx, error_type,
              det[0]-det[2]/2., det[1]-det[3]/2.,
              det[0]+det[2]/2., det[1]+det[3]/2.,
              self._classes[int(det[4])], 
              score
          )
      )

    # load detections
    self._det_rois = {}
    for idx in self._image_idx:
      det_file_name = os.path.join(detection_file_dir, idx+'.txt')
      with open(det_file_name) as f:
        lines = f.readlines()
      f.close()
      bboxes = []
      for line in lines:
        obj = line.strip().split(' ')
        cls = self._class_to_idx[obj[0].lower().strip()]
        xmin = float(obj[4])
        ymin = float(obj[5])
        xmax = float(obj[6])
        ymax = float(obj[7])
        score = float(obj[-1])

        x, y, w, h = bbox_transform_inv([xmin, ymin, xmax, ymax])
        bboxes.append([x, y, w, h, cls, score])
      bboxes.sort(key=lambda x: x[-1], reverse=True)
      self._det_rois[idx] = bboxes

    # do error analysis
    num_dets = 0.
    num_correct = 0.
    num_loc_error = 0.
    num_cls_error = 0.
    num_bg_error = 0.
    num_repeated_error = 0.
    num_detected_obj = 0.

    with open(det_error_file, 'w') as f:
      for idx in self._image_idx:
        gt_bboxes = np.array(self._rois[idx])
        num_objs += len(gt_bboxes)
        detected = [False]*len(gt_bboxes)

        det_bboxes = self._det_rois[idx]
        if len(gt_bboxes) < 1:
          continue

        for i, det in enumerate(det_bboxes):
          if i < len(gt_bboxes):
            num_dets += 1
          ious = batch_iou(gt_bboxes[:, :4], det[:4])
          max_iou = np.max(ious)
          gt_idx = np.argmax(ious)
          if max_iou > 0.1:
            if gt_bboxes[gt_idx, 4] == det[4]:
              if max_iou >= 0.5:
                if i < len(gt_bboxes):
                  if not detected[gt_idx]:
                    num_correct += 1
                    detected[gt_idx] = True
                  else:
                    num_repeated_error += 1
              else:
                if i < len(gt_bboxes):
                  num_loc_error += 1
                  _save_detection(f, idx, 'loc', det, det[5])
            else:
              if i < len(gt_bboxes):
                num_cls_error += 1
                _save_detection(f, idx, 'cls', det, det[5])
          else:
            if i < len(gt_bboxes):
              num_bg_error += 1
              _save_detection(f, idx, 'bg', det, det[5])

        for i, gt in enumerate(gt_bboxes):
          if not detected[i]:
            _save_detection(f, idx, 'missed', gt, -1.0)
        num_detected_obj += sum(detected)
    f.close()

    print ('Detection Analysis:')
    print ('    Number of detections: {}'.format(num_dets))
    print ('    Number of objects: {}'.format(num_objs))
    print ('    Percentage of correct detections: {}'.format(
      num_correct/num_dets))
    print ('    Percentage of localization error: {}'.format(
      num_loc_error/num_dets))
    print ('    Percentage of classification error: {}'.format(
      num_cls_error/num_dets))
    print ('    Percentage of background error: {}'.format(
      num_bg_error/num_dets))
    print ('    Percentage of repeated detections: {}'.format(
      num_repeated_error/num_dets))
    print ('    Recall: {}'.format(
      num_detected_obj/num_objs))

    out = {}
    out['num of detections'] = num_dets
    out['num of objects'] = num_objs
    out['% correct detections'] = num_correct/num_dets
    out['% localization error'] = num_loc_error/num_dets
    out['% classification error'] = num_cls_error/num_dets
    out['% background error'] = num_bg_error/num_dets
    out['% repeated error'] = num_repeated_error/num_dets
    out['% recall'] = num_detected_obj/num_objs

    return out
#mc = kitti_squeezeDet_config()
#x = kitti_gcp(image_set='train',bucket_name='softnautics_dataset_bucket',data_path='kitti/',json_path='/home/shubham/Downloads/lattice-sensai-studio-a7bb94d0d620.json', mc=mc)
#image_per_batch, label_per_batch, box_delta_per_batch, aidx_per_batch,bbox_per_batch, image_per_batch_viz = x.read_batch()

#print(len(image_per_batch))

