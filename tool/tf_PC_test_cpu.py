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

def py_nms(roi,nms_thresh):
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

    def __call__(self, loc, score,
                 anchor, img_size, train=True, scale=1.):
        if train:
            n_pre_nms = self.n_train_pre_nms
            n_post_nms = self.n_train_post_nms
        else:
            n_pre_nms = self.n_test_pre_nms
            n_post_nms = self.n_test_post_nms
        print('======================',n_pre_nms,n_post_nms,self.min_size,self.nms_thresh,'==============================')
        h, w = img_size
        h=tf.to_float(h)
        w=tf.to_float(w)
        roi = loc2bbox(loc,anchor)

        roi=tf.clip_by_value(roi,[0,0,0,0],[h,w,h,w])
        hw=roi[:,2:4]-roi[:,:2]
        inds=tf.reduce_all(hw>=self.min_size,axis=-1)


        # y1, x1, y2, x2 = tf.split(roi, 4, axis=-1)
        # y1 = tf.clip_by_value(y1, 0, h)
        # y2 = tf.clip_by_value(y2, 0, h)
        # x1 = tf.clip_by_value(x1, 0, w)
        # x2 = tf.clip_by_value(x2, 0, w)
        # h = y2 - y1
        # w = x2 - x1
        # roi = tf.concat([y1, x1, y2, x2], axis=1)
        # inds = (h >= self.min_size) & (w >= self.min_size)


        inds=tf.reshape(inds,(-1,))
        roi=tf.boolean_mask(roi,inds)
        score=tf.boolean_mask(score,inds)

        score, top_k = tf.nn.top_k(score, k=tf.reduce_min([n_pre_nms, tf.shape(score)[0]]))
        roi=tf.gather(roi,top_k)
        inds=tf.image.non_max_suppression(roi,score,n_post_nms,iou_threshold=self.nms_thresh)
        # inds=tf.py_func(py_nms,[roi,self.nms_thresh],tf.int32)[:n_post_nms]
        roi=tf.gather(roi,inds)
        return roi