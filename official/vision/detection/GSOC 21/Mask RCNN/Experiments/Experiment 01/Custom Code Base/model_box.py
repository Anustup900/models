

import numpy as np
from collections import namedtuple
import tensorflow as tf

from tensorpack.tfutils.scope_utils import under_name_scope

from config import config

@under_name_scope()
def clip_boxes_batch(boxes, window, batch_ids, name=None):
    """
    Args:
        boxes: nx(#class)x4, xyxy
        window: BSx2 
        batch_ids: 1-D Tensor of size n
    """
    boxes = tf.maximum(boxes, 0.0)

    wh = tf.reverse(tf.gather(window, batch_ids), axis=[1]) 
    whwh = tf.concat((wh, wh), axis=1)

    multiples = tf.stack([tf.constant(1), tf.shape(boxes)[1], tf.constant(1)])
    whwh_tiled = tf.tile(tf.expand_dims(whwh, 1), multiples)    # asd = [1, #class, 1]

    boxes = tf.minimum(boxes, tf.cast(whwh_tiled, tf.float32), name=name)
    return boxes



@under_name_scope()
def clip_boxes(boxes, window, name=None):
    """
    Args:
        boxes: nx4, xyxy
        window: [h, w]
    """
    window = tf.squeeze(window, axis=0)

    boxes = tf.maximum(boxes, 0.0)
    m = tf.tile(tf.reverse(window, [0]), [2])    # (4,)
    boxes = tf.minimum(boxes, tf.cast(m, tf.float32), name=name)
    return boxes


@under_name_scope()
def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: (..., 4), logits
        anchors: (..., 4), floatbox. Must have the same shape

    Returns:
        box_decoded: (..., 4), float32. With the same shape.
    """
    orig_shape = tf.shape(anchors)
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (...)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    clip = np.log(config.PREPROC.MAX_SIZE / 16.)
    wbhb = tf.exp(tf.minimum(box_pred_twth, clip)) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5    # (...)x1x2
    out = tf.concat([x1y1, x2y2], axis=-2)
    return tf.reshape(out, orig_shape)


@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Args:
        boxes: (..., 4), float32
        anchors: (..., 4), float32

    Returns:
        box_encoded: (..., 4), float32 with the same shape.
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    waha = anchors_x2y2 - anchors_x1y1
    xaya = (anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
    wbhb = boxes_x2y2 - boxes_x1y1
    xbyb = (boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero
    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
    return tf.reshape(encoded, tf.shape(boxes))






@under_name_scope()
def crop_and_resize(image, boxes, box_ind, crop_size, orig_image_dims, pad_border=True, verbose_batch_index=None):
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
    prefix = "custom crop_and_resize"
    if verbose_batch_index is not None:
        prefix += f' B_IDX {verbose_batch_index}'

    assert isinstance(crop_size, int), crop_size


    # boxes = print_runtime_shape("boxes (early)", boxes, prefix=prefix)

    boxes = tf.stop_gradient(boxes)

#    org_img_size = tf.concat((tf.constant([-1, -1], dtype=tf.int32), orig_image_dims), axis=0)
#    image = tf.slice(image, begin=tf.zeros(4, dtype=tf.int32), size=org_img_size)
    image = image[:, :, :orig_image_dims[0], :orig_image_dims[1]]

    # TF's crop_and_resize produces zeros on border
    if pad_border:
        # this can be quite slow
        image = tf.pad(image, [[0, 0], [0, 0], [1, 1], [1, 1]], mode='SYMMETRIC')
        boxes = boxes + 1

    @under_name_scope()
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
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.cast(crop_shape[1], tf.float32)
        spacing_h = (y1 - y0) / tf.cast(crop_shape[0], tf.float32)

        imshape = [tf.cast(image_shape[0] - 1, tf.float32), tf.cast(image_shape[1] - 1, tf.float32)]
        nx0 = (x0 + spacing_w / 2 - 0.5) / imshape[1]
        ny0 = (y0 + spacing_h / 2 - 0.5) / imshape[0]

        nw = spacing_w * tf.cast(crop_shape[1] - 1, tf.float32) / imshape[1]
        nh = spacing_h * tf.cast(crop_shape[0] - 1, tf.float32) / imshape[0]

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    # Expand bbox to a minium size of 1
    # boxes_x1y1, boxes_x2y2 = tf.split(boxes, 2, axis=1)
    # boxes_wh = boxes_x2y2 - boxes_x1y1
    # boxes_center = tf.reshape((boxes_x2y2 + boxes_x1y1) * 0.5, [-1, 2])
    # boxes_newwh = tf.maximum(boxes_wh, 1.)
    # boxes_x1y1new = boxes_center - boxes_newwh * 0.5
    # boxes_x2y2new = boxes_center + boxes_newwh * 0.5
    # boxes = tf.concat([boxes_x1y1new, boxes_x2y2new], axis=1)



    boxes = transform_fpcoor_for_tf(boxes, orig_image_dims, [crop_size, crop_size])
    image = tf.transpose(image, [0, 2, 3, 1])   # nhwc

    # image = print_runtime_shape("image", image, prefix=prefix)
    # boxes = print_runtime_shape("boxes (late)", boxes, prefix=prefix)
    ret = tf.image.crop_and_resize(
        image, boxes, tf.cast(box_ind, tf.int32),
        crop_size=[crop_size, crop_size])
    ret = tf.transpose(ret, [0, 3, 1, 2])   # ncss
    return ret



class RPNAnchors(namedtuple('_RPNAnchors', ['boxes', 'gt_labels', 'gt_boxes'])):
    """
    boxes (FS x FS x NA x 4): The anchor boxes.
    gt_labels (FS x FS x NA):
    gt_boxes (FS x FS x NA x 4): Groundtruth boxes corresponding to each anchor.
    """
    def encoded_gt_boxes(self):
        return encode_bbox_target(self.gt_boxes, self.boxes)

    def decode_logits(self, logits):
        return decode_bbox_target(logits, self.boxes)

    @under_name_scope()
    def narrow_to_featuremap_dims(self, featuremap_dims):
        """
        Take in h and w of featuremap that we need to narrow to.
        Separate batch implementation because we need to handle padding.
        # Boxes don't have a batch_dim when this function is called - and neither should
        """

        slice3d = tf.concat([featuremap_dims, [-1]], axis=0)
        slice4d = tf.concat([featuremap_dims, [-1, -1]], axis=0)

        boxes = tf.slice(self.boxes, [0, 0, 0, 0], slice4d)
        gt_labels = tf.slice(self.gt_labels, [0, 0, 0], slice3d)
        gt_boxes = tf.slice(self.gt_boxes, [0, 0, 0, 0], slice4d)
        return RPNAnchors(boxes, gt_labels, gt_boxes)


if __name__ == '__main__':
    """
    Demonstrate what's wrong with tf.image.crop_and_resize:
    """
    import tensorflow.contrib.eager as tfe
    tfe.enable_eager_execution()

    # want to crop 2x2 out of a 5x5 image, and resize to 4x4
    image = np.arange(25).astype('float32').reshape(5, 5)
    boxes = np.asarray([[1, 1, 3, 3]], dtype='float32')
    target = 4

    print(crop_and_resize(
        image[None, None, :, :], boxes, [0], target)[0][0])
    """
    Expected values:
    4.5 5 5.5 6
    7 7.5 8 8.5
    9.5 10 10.5 11
    12 12.5 13 13.5

    You cannot easily get the above results with tf.image.crop_and_resize.
    Try out yourself here:
    """
    print(tf.image.crop_and_resize(
        image[None, :, :, None],
        np.asarray([[1, 1, 2, 2]]) / 4.0, [0], [target, target])[0][:, :, 0])
