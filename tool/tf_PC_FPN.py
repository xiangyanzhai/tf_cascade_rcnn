# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf

from chainer.backends import cuda
from chainercv.utils.bbox.non_maximum_suppression import \
    non_maximum_suppression


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    yx = pre_loc[..., :2] * c_hw + c_yx
    hw = tf.exp(pre_loc[..., 2:4]) * c_hw
    yx1 = yx - hw / 2
    yx2 = yx + hw / 2
    bboxes = tf.concat((yx1, yx2), axis=-1)
    return bboxes


def py_nms(roi, nms_thresh):
    keep = non_maximum_suppression(
        cuda.to_gpu(roi),
        thresh=nms_thresh)
    keep = cuda.to_cpu(keep)
    return keep


class ProposalCreator(object):
    def __init__(self,
                 nms_thresh=0.7,
                 n_train_pre_nms=12000,
                 n_train_post_nms=2000,
                 n_test_pre_nms=6000,
                 n_test_post_nms=300,
                 force_cpu_nms=False,
                 min_size=16,
                 ):

        self.nms_thresh = nms_thresh
        self.n_train_pre_nms = n_train_pre_nms
        self.n_train_post_nms = n_train_post_nms
        self.n_test_pre_nms = n_test_pre_nms
        self.n_test_post_nms = n_test_post_nms
        self.force_cpu_nms = force_cpu_nms
        self.min_size = min_size

    # def __call__(self, loc, score,
    #              anchor, img_size, map_HW, train=True, scale=1.):
    #     if train:
    #         n_pre_nms = self.n_train_pre_nms
    #         n_post_nms = self.n_train_post_nms
    #     else:
    #         n_pre_nms = self.n_test_pre_nms
    #         n_post_nms = self.n_test_post_nms
    #     print('======================', n_pre_nms, n_post_nms, self.min_size, self.nms_thresh,
    #           '==============================')
    #     h, w = img_size
    #     h = tf.to_float(h)
    #     w = tf.to_float(w)
    #     roi = loc2bbox(loc, anchor)
    #
    #     roi = tf.clip_by_value(roi, [0, 0, 0, 0], [h, w, h, w])
    #     # hw = roi[:, 2:4] - roi[:, :2]
    #     # inds = tf.reduce_all(hw >= self.min_size, axis=-1)
    #
    #     Roi = []
    #     Score = []
    #     C = 0
    #     for i in range(5):
    #         map_H, map_W = map_HW[i]
    #         c = map_H * map_W * 3
    #         tscore=score[C:c]
    #         troi=roi[C:c]
    #         C+=c
    #         hw=troi[:,2:4]-troi[:,:2]
    #         pass
    #
    #     inds = tf.reshape(inds, (-1,))
    #     roi = tf.boolean_mask(roi, inds)
    #     score = tf.boolean_mask(score, inds)
    #
    #     score, top_k = tf.nn.top_k(score, k=tf.reduce_min([n_pre_nms, tf.shape(score)[0]]))
    #     roi = tf.gather(roi, top_k)
    #     # inds=tf.image.non_max_suppression(roi,score,n_post_nms,iou_threshold=self.nms_thresh)
    #     inds = tf.py_func(py_nms, [roi, self.nms_thresh], tf.int32)[:n_post_nms]
    #     roi = tf.gather(roi, inds)
    #     return roi

    def __call__(self, loc, score,
                 anchor, img_size, map_HW, train=True, scale=1.):
        if train:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms

        # self.min_size = [4, 8, 16, 32, 64]
        print('======================', n_pre_nms, n_post_nms, self.min_size, self.nms_thresh,
              '==============================')
        h, w = img_size
        h = tf.to_float(h)
        w = tf.to_float(w)
        roi = loc2bbox(loc, anchor)
        roi = tf.clip_by_value(roi, [0, 0, 0, 0], [h, w, h, w])

        Roi = []
        Score = []
        C = 0

        # self.min_size = [4, 4, 4, 4, 4]
        for i in range(5):
            map_H, map_W = map_HW[i]
            c = map_H * map_W * 3
            tscore = score[C:C + c]
            troi = roi[C:C + c]
            C += c
            hw = troi[:, 2:4] - troi[:, :2]
            inds = tf.reduce_all(hw >= self.min_size[i], axis=1)
            inds = tf.reshape(inds, (-1,))
            troi = tf.boolean_mask(troi, inds)
            tscore = tf.boolean_mask(tscore, inds)
            tscore, top_k = tf.nn.top_k(tscore, k=tf.reduce_min([n_post_nms, tf.shape(tscore)[0]]))
            troi = tf.gather(troi, top_k)
            Roi.append(troi)
            Score.append(tscore)

        roi = tf.concat(Roi, axis=0)
        score = tf.concat(Score, axis=0)

        score, top_k = tf.nn.top_k(score, k=tf.shape(score)[0])
        roi = tf.gather(roi, top_k)
        # inds=tf.image.non_max_suppression(roi,score,n_post_nms,iou_threshold=self.nms_thresh)
        inds = tf.py_func(py_nms, [roi, self.nms_thresh], tf.int32)[:n_post_nms]
        inds = tf.reshape(inds, (-1,))
        roi = tf.gather(roi, inds)

        return roi
