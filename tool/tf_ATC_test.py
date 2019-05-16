# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as  tf


def cal_IOU(pre_bboxes, bboxes):
    hw = pre_bboxes[:, 2:4] - pre_bboxes[:, :2]
    areas1 = tf.reduce_prod(hw, axis=-1)

    hw = bboxes[:, 2:4] - bboxes[:, :2]
    areas2 = tf.reduce_prod(hw, axis=-1)

    yx1 = tf.maximum(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = tf.minimum(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])

    hw = yx2 - yx1
    hw = tf.maximum(hw, 0)
    areas_i = tf.reduce_prod(hw, axis=-1)
    iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    return iou


def bbox2loc(anchor, bbox):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    hw = bbox[..., 2:4] - bbox[..., 0:2]
    yx = bbox[..., :2] + hw / 2
    t_yx = (yx - c_yx) / c_hw
    t_hw = tf.log(hw / c_hw)
    return tf.concat([t_yx, t_hw], axis=1)


class AnchorTargetCreator(object):
    def __init__(self,
                 n_sample=256,
                 pos_iou_thresh=0.7, neg_iou_thresh=0.3,
                 pos_ratio=0.5):
        self.n_sample = n_sample
        self.pos_iou_thresh = pos_iou_thresh
        self.neg_iou_thresh = neg_iou_thresh
        self.pos_ratio = pos_ratio

    def __call__(self, bbox, anchor, img_size):
        return tf.cond(tf.equal(tf.shape(bbox)[0], 0), lambda: self.func1(anchor),
                       lambda: self.func2(bbox, anchor, img_size))

    def func1(self, anchor):
        inds = tf.range(tf.shape(anchor)[0])
        inds = tf.random_shuffle(inds)[:self.n_sample]
        label = tf.zeros(tf.shape(inds), dtype=tf.int32)
        indsP = tf.zeros([0], dtype=tf.int64)
        loc = tf.zeros([0, 4], dtype=tf.float32)
        inds = tf.to_int64(inds)
        return inds, label, indsP, loc
    def func2(self, bbox, anchor, img_size):
        h, w = img_size
        index_inside = (anchor[:, 0] >= 0.0) & (anchor[:, 1] >= 0.0) & (anchor[:, 2] <= tf.to_float(h)) & (
                anchor[:, 3] <= tf.to_float(w))
        index_inside = tf.where(index_inside)[:, 0]
        anchor = tf.gather(anchor, index_inside)

        IOU = cal_IOU(anchor, bbox)

        inds_box = tf.argmax(IOU, axis=1, output_type=tf.int32)

        inds = tf.range(tf.shape(anchor)[0])

        inds = tf.concat([tf.reshape(inds, (-1, 1)), tf.reshape(inds_box, (-1, 1))], axis=1)
        iou = tf.gather_nd(IOU, inds)

        indsP1 = iou >= self.pos_iou_thresh
        indsN = iou < self.neg_iou_thresh

        t = tf.reduce_max(IOU, axis=0)
        t = tf.equal(IOU, t)
        indsP2 = tf.reduce_any(t, axis=1)

        inds_gt_box = tf.argmax(tf.to_int32(t), axis=1, output_type=tf.int32)
        inds_box = inds_box * tf.to_int32(~indsP2) + inds_gt_box

        indsP = indsP1 | indsP2

        if False:
            indsN = indsN & (~indsP2)
        else:
            indsP = indsP & (~indsN)
            print('注意：***************这里是个参数*************  tf_ATC_test.py ')

        indsP = tf.where(indsP)[:, 0]
        indsN = tf.where(indsN)[:, 0]
        indsP = tf.random_shuffle(indsP)
        indsN = tf.random_shuffle(indsN)

        n_pos = tf.reduce_min([tf.to_int32(self.n_sample * self.pos_ratio), tf.shape(indsP)[0]])
        indsP = indsP[:n_pos]
        indsN = indsN[:self.n_sample - n_pos]

        anchor = tf.gather(anchor, indsP)
        bbox = tf.gather(bbox, tf.gather(inds_box, indsP))
        loc = bbox2loc(anchor, bbox)

        label = tf.concat(
            [tf.ones(dtype=tf.int32, shape=[n_pos]), tf.zeros(dtype=tf.int32, shape=[tf.shape(indsN)[0]])],
            axis=0)
        indsP = tf.gather(index_inside, indsP)
        indsN = tf.gather(index_inside, indsN)
        inds = tf.concat([indsP, indsN], axis=0)
        return inds, label, indsP, loc
