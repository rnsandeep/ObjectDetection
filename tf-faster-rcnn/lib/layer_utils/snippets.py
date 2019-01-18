# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
from layer_utils.generate_anchors import generate_anchors

def generate_anchors_pre(height, width, feat_stride, anchor_scales=(8,16,32), anchor_ratios=(0.5,1,2)):
  """ A wrapper function to generate anchors given different scales
    Also return the number of anchors in variable 'length'
  """
  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  shift_x = np.arange(0, width) * feat_stride
  shift_y = np.arange(0, height) * feat_stride
  shift_x, shift_y = np.meshgrid(shift_x, shift_y)
  shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
  K = shifts.shape[0]
  # width changes faster, so here it is H, W, C
  anchors = anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2))
  anchors = anchors.reshape((K * A, 4)).astype(np.float32, copy=False)
  length = np.int32(anchors.shape[0])

  return anchors, length

def generate_anchors_pre_tf(height, width, feat_stride=16, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
  shift_x = tf.range(width) * feat_stride # width
  shift_y = tf.range(height) * feat_stride # height
  shift_x, shift_y = tf.meshgrid(shift_x, shift_y)
  sx = tf.reshape(shift_x, shape=(-1,))
  sy = tf.reshape(shift_y, shape=(-1,))
  shifts = tf.transpose(tf.stack([sx, sy, sx, sy]))
  K = tf.multiply(width, height)
  shifts = tf.transpose(tf.reshape(shifts, shape=[1, K, 4]), perm=(1, 0, 2))

  anchors = generate_anchors(ratios=np.array(anchor_ratios), scales=np.array(anchor_scales))
  A = anchors.shape[0]
  anchor_constant = tf.constant(anchors.reshape((1, A, 4)), dtype=tf.int32)

  length = K * A
  anchors_tf = tf.reshape(tf.add(anchor_constant, shifts), shape=(length, 4))
  
  return tf.cast(anchors_tf, dtype=tf.float32), length

def roi_align(featuremap, boxes, resolution):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        resolution: output spatial resolution
    Returns:
        NxCx res x res
    """
    # sample 4 locations per roi bin
    batch_ids = tf.squeeze(tf.slice(boxes, [0, 0], [-1, 1], name="batch_id"), [1])
    ret = crop_and_resize(
        featuremap, boxes,
        tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        resolution )
    return ret

def crop_and_resize(image, boxes, box_ind, crop_size, pad_border=True):
    """
    Aligned version of tf.image.crop_and_resize, following our definition of floating point boxes.
    Args:
        image: NCHW
        boxes: nx4, x1y1x2y2
        box_ind: (n,)
        crop_size (int):
    Returns:
        n,C,size,size
    """
    assert isinstance(crop_size, int), crop_size
    boxes = tf.stop_gradient(boxes)

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        boxes = boxes + 1

    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way tf.image.crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.
        However, what we want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample (in my definition) assumes floating point coordinate
         (0.0, 0.0) is the same as pixel value (0, 0))
        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize
        Returns:
            y1x1y2x2
        """
        z, x0, y0, x1, y1 = tf.split(boxes, 5, axis=1)

        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2 - 0.5) / (tf.to_float(image_shape[1] - 1))
        ny0 = (y0 + spacing_h / 2 - 0.5) / (tf.to_float(image_shape[0] - 1))

        nw = spacing_w * tf.to_float(crop_shape[1] - 1) / (tf.to_float(image_shape[1] - 1))
        nh = spacing_h * tf.to_float(crop_shape[0] - 1) / (tf.to_float(image_shape[0] - 1))

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # Expand bbox to a minium size of 1
    # boxes_x1y1, boxes_x2y2 = tf.split(boxes, 2, axis=1)
    # boxes_wh = boxes_x2y2 - boxes_x1y1
    # boxes_center = tf.reshape((boxes_x2y2 + boxes_x1y1) * 0.5, [-1, 2])
    # boxes_newwh = tf.maximum(boxes_wh, 1.)
    # boxes_x1y1new = boxes_center - boxes_newwh * 0.5
    # boxes_x2y2new = boxes_center + boxes_newwh * 0.5
    # boxes = tf.concat([boxes_x1y1new, boxes_x2y2new], axis=1)

    image_shape = tf.shape(image)[2:]
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])   # nhwc
    ret = tf.image.crop_and_resize(
        image, boxes, tf.to_int32(box_ind),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret

