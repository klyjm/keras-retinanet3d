"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from . import backend
import tensorflow as tf
from keras import backend as K
import math


def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.

    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.

    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.

        As defined in https://arxiv.org/abs/1708.02002

        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).

        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        labels         = y_true[:, :, :-1]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = backend.where(keras.backend.not_equal(anchor_state, -1))
        labels         = backend.gather_nd(labels, indices)
        classification = backend.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = backend.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = backend.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = backend.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.

    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.

    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.

        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).

        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        regression        = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = backend.where(keras.backend.equal(anchor_state, 1))
        regression        = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(keras.backend.stack([regression_diff[:, 0] + regression_diff[:, 2], regression_diff[:, 1] + regression_diff[:, 3]], axis=1))
        # regression_diff = keras.backend.abs(regression_diff)
        regression_loss = backend.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1


def diou_loss():
    def _diou(y_true, y_pred):
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        #
        # # filter out "ignore" anchors
        indices = backend.where(keras.backend.equal(anchor_state, 1))
        regression = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)
        # boxes1 = regression
        # boxes2 = regression_target[..., :4]
        # boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        # boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
        #
        # left_up = tf.maximum(boxes1[..., :2], boxes2[..., :2])
        # right_down = tf.minimum(boxes1[..., 2:], boxes2[..., 2:])
        #
        # inter_section = tf.maximum(right_down - left_up, 0.0)
        # inter_area = inter_section[..., 0] * inter_section[..., 1]
        # union_area = boxes1_area + boxes2_area - inter_area
        # iou = inter_area / union_area
        #
        # enclose_left_up = tf.minimum(boxes1[..., :2], boxes2[..., :2])
        # enclose_right_down = tf.maximum(boxes1[..., 2:], boxes2[..., 2:])
        # enclose = tf.maximum(enclose_right_down - enclose_left_up, 0.0)
        # enclose_area = enclose[..., 0] * enclose[..., 1]
        # giou = iou - 1.0 * (enclose_area - union_area) / enclose_area
        # separate target and state
        # regression = y_pred[:, :, :4]
        # y_pred_iou = y_pred[:, :, 4]
        # y_pred_hard_scores = y_pred[:, :, 5]
        # regression_target = y_true[:, :, :4]
        # anchor_state = y_true[:, :, 4]
        #
        # # filter out "ignore" anchors
        # indices = backend.where(
        #     (keras.backend.equal(anchor_state, 1) & (keras.backend.greater(y_pred_hard_scores, 0.1))))
        # regression = backend.gather_nd(regression, indices)
        # y_pred_iou = backend.gather_nd(y_pred_iou, indices)
        # y_pred_iou = keras.backend.expand_dims(y_pred_iou)
        #
        # regression_target = backend.gather_nd(regression_target, indices)
        #
        # y_true_iou = intersection_over_union(regression_target, regression)
        # iou_loss = keras.backend.binary_crossentropy(output=y_pred_iou, target=y_true_iou)
        pred_left = tf.minimum(regression[:, 0], regression[:, 2])
        pred_top = tf.maximum(regression[:, 3], regression[:, 1])
        pred_right = tf.maximum(regression[:, 0], regression[:, 2])
        pred_bottom = tf.minimum(regression[:, 3], regression[:, 1])

        # (num_pos, )
        target_left = tf.minimum(regression_target[:, 0], regression_target[:, 2])
        target_top = tf.maximum(regression_target[:, 3], regression_target[:, 1])
        target_right = tf.maximum(regression_target[:, 0], regression_target[:, 2])
        target_bottom = tf.minimum(regression_target[:, 3], regression_target[:, 1])

        target_area = K.abs(target_left - target_right) * K.abs(target_top - target_bottom)
        pred_area = K.abs(pred_left - pred_right) * K.abs(pred_top - pred_bottom)
        w_intersect = tf.maximum(- tf.maximum(pred_left, target_left) + tf.minimum(pred_right, target_right), 0.0)
        h_intersect = tf.maximum(- tf.maximum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top), 0.0)
        w_outersect = - tf.minimum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        h_outersect = - tf.minimum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
        target_center_x = (target_right + target_left) / 2
        target_center_y = (target_top + target_bottom) / 2
        pred_center_x = (pred_right + pred_left) / 2
        pred_center_y = (pred_top + pred_bottom) / 2

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        c = K.pow(w_outersect, 2) + K.pow(h_outersect, 2)
        rou = K.pow((target_center_x - pred_center_x), 2) + K.pow((target_center_y - pred_center_y), 2)

        # (num_pos, )
        # iou_loss = - tf.log((area_intersect + 1e-7) / (area_union + 1e-7))
        iou_loss = 1 - (area_intersect + 1e-7) / (area_union + 1e-7) + rou / c
        # iou_loss = - tf.log((area_intersect + 1e-7) / (area_union + 1e-7)) + rou / c

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(iou_loss) / normalizer

        # enclose_left_up = tf.minimum(boxes1_x0y0x1y1[..., :2], boxes2_x0y0x1y1[..., :2])
        # enclose_right_down = tf.maximum(boxes1_x0y0x1y1[..., 2:], boxes2_x0y0x1y1[..., 2:])
        #
        # enclose_wh = enclose_right_down - enclose_left_up
        # enclose_c2 = K.pow(enclose_wh[..., 0], 2) + K.pow(enclose_wh[..., 1], 2)
        #
        # p2 = K.pow(boxes1[..., 0] - boxes2[..., 0], 2) + K.pow(boxes1[..., 1] - boxes2[..., 1], 2)
        #
        # diou = iou - 1.0 * p2 / enclose_c2
        #
        # return diou
    return _diou



def ciou_loss():
    def _ciou(y_true, y_pred):
        regression = y_pred
        regression_target = y_true[:, :, :-1]
        anchor_state = y_true[:, :, -1]
        #
        # # filter out "ignore" anchors
        indices = backend.where(keras.backend.equal(anchor_state, 1))
        regression = backend.gather_nd(regression, indices)
        regression_target = backend.gather_nd(regression_target, indices)
        pred_left = tf.minimum(regression[:, 0], regression[:, 2])
        pred_top = tf.maximum(regression[:, 3], regression[:, 1])
        pred_right = tf.maximum(regression[:, 0], regression[:, 2])
        pred_bottom = tf.minimum(regression[:, 3], regression[:, 1])

        # (num_pos, )
        target_left = tf.minimum(regression_target[:, 0], regression_target[:, 2])
        target_top = tf.maximum(regression_target[:, 3], regression_target[:, 1])
        target_right = tf.maximum(regression_target[:, 0], regression_target[:, 2])
        target_bottom = tf.minimum(regression_target[:, 3], regression_target[:, 1])

        w_gt = K.abs(target_left - target_right)
        h_gt = K.abs(target_top - target_bottom)
        w = K.abs(pred_left - pred_right)
        h = K.abs(pred_top - pred_bottom)
        target_area = w_gt * h_gt
        pred_area = w * h
        w_intersect = tf.maximum(- tf.maximum(pred_left, target_left) + tf.minimum(pred_right, target_right), 0.0)
        h_intersect = tf.maximum(- tf.maximum(pred_bottom, target_bottom) + tf.minimum(pred_top, target_top), 0.0)
        w_outersect = - tf.minimum(pred_left, target_left) + tf.maximum(pred_right, target_right)
        h_outersect = - tf.minimum(pred_bottom, target_bottom) + tf.maximum(pred_top, target_top)
        target_center_x = (target_right + target_left) / 2
        target_center_y = (target_top + target_bottom) / 2
        pred_center_x = (pred_right + pred_left) / 2
        pred_center_y = (pred_top + pred_bottom) / 2

        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        c = K.pow(w_outersect, 2) + K.pow(h_outersect, 2)
        rou = K.pow((target_center_x - pred_center_x), 2) + K.pow((target_center_y - pred_center_y), 2)

        # (num_pos, )
        # iou_loss = - tf.log((area_intersect + 1e-7) / (area_union + 1e-7))
        v = 4 / (math.pi ** 2) * K.pow(tf.atan(w_gt / h_gt) - tf.atan(w / h), 2)
        iou_loss = 1 - (area_intersect + 1e-7) / (area_union + 1e-7)
        alpha = v / (iou_loss + v)
        ciou = iou_loss + rou / c + alpha * v
        # iou_loss = - tf.log((area_intersect + 1e-7) / (area_union + 1e-7)) + rou / c

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(ciou) / normalizer

    return _ciou


def intersection_over_union(y_true_masks, y_pred_masks):
    w_true = y_true_masks[:, 2::4] - y_true_masks[:, 0::4]
    h_true = y_true_masks[:, 3::4] - y_true_masks[:, 1::4]
    gt_area = w_true * h_true

    w_pred = y_pred_masks[:, 2::4] - y_pred_masks[:, 0::4]
    h_pred = y_pred_masks[:, 3::4] - y_pred_masks[:, 1::4]
    pred_area = w_pred * h_pred
    w_intersection = keras.backend.maximum(0., keras.backend.minimum(y_true_masks[:, 2::4],
                                                                     y_pred_masks[:, 2::4]) - keras.backend.maximum(
        y_true_masks[:, 0::4], y_pred_masks[:, 0::4]))
    h_intersection = keras.backend.maximum(0., keras.backend.minimum(y_true_masks[:, 3::4],
                                                                     y_pred_masks[:, 3::4]) - keras.backend.maximum(
        y_true_masks[:, 1::4], y_pred_masks[:, 1::4]))
    intersection_area = w_intersection * h_intersection

    union = pred_area + gt_area - intersection_area + keras.backend.epsilon()
    return intersection_area / union