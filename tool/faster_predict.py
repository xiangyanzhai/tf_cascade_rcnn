# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as tf

iou_thresh = None


def loc2bbox(pre_loc, anchor):
    c_hw = anchor[..., 2:4] - anchor[..., 0:2]
    c_yx = anchor[..., :2] + c_hw / 2
    yx = pre_loc[..., :2] * c_hw + c_yx
    hw = tf.exp(pre_loc[..., 2:4]) * c_hw
    yx1 = yx - hw / 2
    yx2 = yx + hw / 2
    bboxes = tf.concat((yx1, yx2), axis=-1)
    return bboxes

def py_inds(score, inds):

    score[inds] = score[inds] * -1
    inds = score > 0
    score[inds] = 0
    score = score * -1

    return score

def fn_map(x):
    bboxes = x[0]
    score = x[1]
    cls = x[2]
    m=tf.shape(score)[0]
    keep=tf.image.non_max_suppression(bboxes,score,m,iou_threshold=iou_thresh)
    score = tf.py_func(py_inds, [score, keep], tf.float32)
    score = tf.reshape(score, (-1, 1))
    cls=tf.zeros(m)+cls
    cls = tf.reshape(cls, (-1, 1))
    pre = tf.concat([bboxes, score, cls], axis=-1)
    return pre



def predict(net_m_loc, net_m_score, roi, img_H, img_W, iou_thresh_=0.3, c_thresh=1e-3):
    # m*cls*4
    # m*cls
    global iou_thresh
    iou_thresh=iou_thresh_
    net_m_loc = tf.transpose(net_m_loc, [1, 0, 2])[1:]
    bboxes = loc2bbox(net_m_loc, roi)

    img_H=tf.to_float(img_H)
    img_W=tf.to_float(img_W)
    bboxes = tf.clip_by_value(bboxes, [0, 0, 0, 0], [img_H, img_W, img_H, img_W])
    net_m_score = tf.nn.softmax(net_m_score)
    net_m_score = tf.transpose(net_m_score)
    net_m_score = net_m_score[1:]

    cls = tf.range(tf.shape(bboxes)[0])
    cls=tf.to_float(cls)
    # print(bboxes,net_m_score,cls)
    pre = tf.map_fn(fn_map, [bboxes, net_m_score, cls], tf.float32)
    pre=tf.reshape(pre,(-1,6))
    pre=tf.boolean_mask(pre,pre[:,-2]>c_thresh)
    _,top_k=tf.nn.top_k(pre[:,-2],tf.shape(pre)[0])
    pre=tf.gather(pre,top_k)
    return pre

