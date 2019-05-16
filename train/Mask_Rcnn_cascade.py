# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from tf_cascade_rcnn.tool.ROIAlign import roi_align
from tf_cascade_rcnn.tool.tf_ATC_FPN import AnchorTargetCreator
from tf_cascade_rcnn.tool.tf_PC_FPN import ProposalCreator
from tf_cascade_rcnn.tool.tf_PTC_mask_test import ProposalTargetCreator
from tf_cascade_rcnn.tool.tf_PTC_test import ProposalTargetCreator as ProposalTargetCreator_box
from tf_cascade_rcnn.tool.read_Data_mask import readData
from tf_cascade_rcnn.tool.get_anchors import get_anchors
from tf_cascade_rcnn.tool import resnet_v1
from tf_cascade_rcnn.tool.config import Config


def abs_smooth(net_loc, input_loc):
    """Smoothed absolute function. Useful to compute an L1 smooth error.

    Define as:
        x^2 / 2         if abs(x) < 1
        abs(x) - 0.5    if abs(x) > 1
    We use here a differentiable definition using min(x) and abs(x). Clearly
    not optimal, but good enough for our purpose!
    """
    x = net_loc - input_loc
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)
    return r


def softmaxloss(score, label):
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=score, labels=label))
    return loss


def SmoothL1Loss(net_loc, input_loc, sigma, num):
    net_loc_yh = tf.abs(net_loc - input_loc)
    part1 = tf.boolean_mask(net_loc_yh, net_loc_yh < 1)
    part2 = tf.boolean_mask(net_loc_yh, net_loc_yh >= 1)
    loss = (tf.reduce_sum((part1 * sigma) ** 2 * 0.5) + tf.reduce_sum(part2 - 0.5 / sigma ** 2)) / num
    return loss


def average_gradients(tower_grads):
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = [tf.expand_dims(g, 0) for g, _ in grad_and_vars]
        grads = tf.concat(grads, 0)
        grad = tf.reduce_mean(grads, 0)
        grad_and_var = (grad, grad_and_vars[0][1])
        # [(grad0, var0),(grad1, var1),...]
        average_grads.append(grad_and_var)
    return average_grads


class Mask_rcnn_resnet_101():
    def __init__(self, config):
        self.config = config
        self.Mean = tf.constant(self.config.Mean, dtype=tf.float32)
        self.anchors = []
        self.num_anchor = []
        for i in range(5):
            self.num_anchor.append(len(config.anchor_scales[i]) * len(config.anchor_ratios[i]))
            stride = 4 * 2 ** i
            print(stride)
            self.anchors.append(get_anchors(np.ceil(self.config.img_max / stride + 1), self.config.anchor_scales[i],
                                            self.config.anchor_ratios[i], stride=stride))

        self.ATC = AnchorTargetCreator(n_sample=config.rpn_n_sample, pos_iou_thresh=config.rpn_pos_iou_thresh,
                                       neg_iou_thresh=config.rpn_neg_iou_thresh, pos_ratio=config.rpn_pos_ratio)
        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms, n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)
        self.PTC_1 = ProposalTargetCreator_box(n_sample=config.fast_n_sample,
                                               pos_ratio=config.fast_pos_ratio,
                                               pos_iou_thresh=config.fast_pos_iou_thresh,
                                               neg_iou_thresh_hi=config.fast_neg_iou_thresh_hi,
                                               neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)
        self.PTC_2 = ProposalTargetCreator_box(n_sample=config.fast_n_sample,
                                               pos_ratio=config.fast_pos_ratio, pos_iou_thresh=0.6,
                                               neg_iou_thresh_hi=0.6,
                                               neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)
        self.PTC = ProposalTargetCreator(n_sample=config.fast_n_sample,
                                         pos_ratio=config.fast_pos_ratio, pos_iou_thresh=0.7,
                                         neg_iou_thresh_hi=0.7,
                                         neg_iou_thresh_lo=config.fast_neg_iou_thresh_lo)
        self.argscope = resnet_v1.resnet_arg_scope(weight_decay=config.weight_decay)
        self.loc_std1 = [1. / 10, 1. / 10, 1. / 5, 1. / 5]
        self.loc_std2 = [1. / 20, 1. / 20, 1. / 10, 1. / 10]
        self.loc_std3 = [1. / 30, 1. / 30, 1. / 15, 1. / 15]
        self.loss_weights = [1.0, 0.5, 0.25]


    def handle_im(self, im, bboxes):
        H = tf.shape(im)[1]
        W = tf.shape(im)[2]
        H = tf.to_float(H)
        W = tf.to_float(W)
        ma = tf.reduce_max([H, W])
        mi = tf.reduce_min([H, W])
        scale = tf.reduce_min([self.config.img_max / ma, self.config.img_min / mi])
        nh = H * scale
        nw = W * scale
        nh = tf.to_int32(nh)
        nw = tf.to_int32(nw)
        im = tf.image.resize_images(im, (nh, nw))

        bboxes = tf.concat([bboxes[..., :4] * scale, bboxes[..., 4:]], axis=-1)
        return im, bboxes, nh, nw

    def fpn_net(self, C):
        C5, C4, C3, C2 = C
        with tf.variable_scope('FPN'):
            with slim.arg_scope([slim.conv2d, ], weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                weights_initializer=tf.variance_scaling_initializer(),
                                activation_fn=None):
                H5 = tf.shape(C5)[1]
                W5 = tf.shape(C5)[2]
                P5 = slim.conv2d(C5, 256, [1, 1], scope='P5_1x1')

                H4 = tf.shape(C4)[1]
                W4 = tf.shape(C4)[2]
                uP5 = tf.image.resize_images(P5, (H4, W4), method=1)
                P4 = slim.conv2d(C4, 256, [1, 1], scope='P4_1x1')
                P4 = P4 + uP5

                H3 = tf.shape(C3)[1]
                W3 = tf.shape(C3)[2]
                uP4 = tf.image.resize_images(P4, (H3, W3), method=1)
                P3 = slim.conv2d(C3, 256, [1, 1], scope='P3_1x1')
                P3 = P3 + uP4

                H2 = tf.shape(C2)[1]
                W2 = tf.shape(C2)[2]
                uP3 = tf.image.resize_images(P3, (H2, W2), method=1)
                P2 = slim.conv2d(C2, 256, [1, 1], scope='P2_1x1')
                P2 = P2 + uP3

                P2 = slim.conv2d(P2, 256, [3, 3], scope='P2_3x3')
                P3 = slim.conv2d(P3, 256, [3, 3], scope='P3_3x3')
                P4 = slim.conv2d(P4, 256, [3, 3], scope='P4_3x3')
                P5 = slim.conv2d(P5, 256, [3, 3], scope='P5_3x3')

                P6 = slim.max_pool2d(P5, [1, 1])
                H6 = tf.shape(P6)[1]
                W6 = tf.shape(P6)[2]

        return [P2, P3, P4, P5, P6], [(H2, W2), (H3, W3), (H4, W4), (H5, W5), (H6, W6)]

    def rpn_net(self, P):
        P2, P3, P4, P5, P6 = P
        a, b, c, d, e = self.num_anchor
        channel = 256
        with tf.variable_scope('rpn') as scope:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
                rpn_P2 = slim.conv2d(P2, channel, [3, 3], scope='conv')
                net_score2 = slim.conv2d(rpn_P2, a * 2, [1, 1], activation_fn=None, scope='cls')
                net_loc2 = slim.conv2d(rpn_P2, a * 4, [1, 1], activation_fn=None, scope='box')
                scope.reuse_variables()

                rpn_P3 = slim.conv2d(P3, channel, [3, 3], scope='conv')
                rpn_P4 = slim.conv2d(P4, channel, [3, 3], scope='conv')
                rpn_P5 = slim.conv2d(P5, channel, [3, 3], scope='conv')
                rpn_P6 = slim.conv2d(P6, channel, [3, 3], scope='conv')

                net_score3 = slim.conv2d(rpn_P3, b * 2, [1, 1], activation_fn=None, scope='cls')
                net_loc3 = slim.conv2d(rpn_P3, b * 4, [1, 1], activation_fn=None, scope='box')

                net_score4 = slim.conv2d(rpn_P4, c * 2, [1, 1], activation_fn=None, scope='cls')
                net_loc4 = slim.conv2d(rpn_P4, c * 4, [1, 1], activation_fn=None, scope='box')

                net_score5 = slim.conv2d(rpn_P5, d * 2, [1, 1], activation_fn=None, scope='cls')
                net_loc5 = slim.conv2d(rpn_P5, d * 4, [1, 1], activation_fn=None, scope='box')

                net_score6 = slim.conv2d(rpn_P6, e * 2, [1, 1], activation_fn=None, scope='cls')
                net_loc6 = slim.conv2d(rpn_P6, e * 4, [1, 1], activation_fn=None, scope='box')
        m = tf.shape(P2)[0]
        net_score2 = tf.reshape(net_score2, (m, -1, 2))
        net_loc2 = tf.reshape(net_loc2, (m, -1, 4))

        net_score3 = tf.reshape(net_score3, (m, -1, 2))
        net_loc3 = tf.reshape(net_loc3, (m, -1, 4))

        net_score4 = tf.reshape(net_score4, (m, -1, 2))
        net_loc4 = tf.reshape(net_loc4, (m, -1, 4))

        net_score5 = tf.reshape(net_score5, (m, -1, 2))
        net_loc5 = tf.reshape(net_loc5, (m, -1, 4))

        net_score6 = tf.reshape(net_score6, (m, -1, 2))
        net_loc6 = tf.reshape(net_loc6, (m, -1, 4))

        net_score = tf.concat([net_score2, net_score3, net_score4, net_score5, net_score6], axis=1)
        net_loc = tf.concat([net_loc2, net_loc3, net_loc4, net_loc5, net_loc6], axis=1)
        return net_score, net_loc

    def fast_train_data_box(self, roi, bboxes, PTC, loc_std):
        roi, loc, label = PTC(roi, bboxes[:, :4], tf.to_int32(bboxes[:, -1]), loc_normalize_std=loc_std)
        area = tf.reduce_prod(roi[:, 2:4] - roi[:, :2] + 1, axis=1)
        roi_inds = tf.floor(4.0 + tf.log(area ** 0.5 / 224.0) / tf.log(2.0))
        roi_inds = tf.clip_by_value(roi_inds, 2, 5)
        roi_inds = roi_inds - 2
        roi_inds = tf.to_int32(roi_inds)
        return roi, roi_inds, loc, label

    def fast_train_data(self, roi, bboxes, masks, img_size, loc_std):

        roi, loc, label, inter, target_inds = self.PTC(roi, bboxes[:, :4], tf.to_int32(bboxes[:, -1]),
                                                       loc_normalize_std=loc_std)

        area = tf.reduce_prod(roi[:, 2:4] - roi[:, :2] + 1, axis=1)
        roi_inds = tf.floor(4.0 + tf.log(area ** 0.5 / 224.0) / tf.log(2.0))
        roi_inds = tf.clip_by_value(roi_inds, 2, 5)
        roi_inds = roi_inds - 2
        roi_inds = tf.to_int32(roi_inds)

        area = tf.reduce_prod(inter[:, 2:4] - inter[:, :2] + 1, axis=1)
        inter_inds = tf.floor(4.0 + tf.log(area ** 0.5 / 224.0) / tf.log(2.0))
        inter_inds = tf.clip_by_value(inter_inds, 2, 5)
        inter_inds = inter_inds - 2
        inter_inds = tf.to_int32(inter_inds)

        img_H, img_W = img_size
        a = tf.concat([[img_H], [img_W], [img_H], [img_W]], axis=0)
        a = tf.to_float(a)
        masks = masks[..., None]

        mask_H = tf.shape(masks)[1]
        mask_W = tf.shape(masks)[2]
        b = tf.concat([[mask_H], [mask_W], [mask_H], [mask_W]], axis=0)
        b = tf.to_float(b)
        inter_norm = inter / a * b
        target = roi_align(masks, inter_norm, target_inds, 28)

        target = target[..., 0]
        return roi, roi_inds, loc, label, inter, inter_inds, target

        pass

    def pooling(self, P, roi, roi_inds, HW, map_HW, x):
        img_H, img_W = HW
        img_H = tf.to_float(img_H)
        img_W = tf.to_float(img_W)
        roi_norm = roi / tf.concat([[img_H], [img_W], [img_H], [img_W]], axis=0)
        scale = tf.concat([map_HW[:4]], axis=0)
        scale = tf.tile(scale, [1, 2])
        scale = tf.gather(scale, roi_inds)
        scale = tf.to_float(scale)

        xx = []
        inds = []
        index = tf.range(tf.shape(roi)[0])
        tinds = tf.zeros(tf.shape(roi)[0], dtype=tf.int32)
        for i in range(4):
            t = tf.equal(roi_inds, i)
            troi = tf.boolean_mask(roi_norm, t) * tf.boolean_mask(scale, t)
            troi = roi_align(P[i], troi, tf.boolean_mask(tinds, t), x)
            xx.append(troi)
            inds.append(tf.boolean_mask(index, t))

        xx = tf.concat(xx, axis=0)
        inds = tf.concat(inds, axis=0)
        _, top_k = tf.nn.top_k(-inds, tf.shape(inds)[0])
        return tf.gather(xx, top_k)

    def fast_net(self, net_fast, scope_name):
        with tf.variable_scope(scope_name):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                net_fast = slim.conv2d(net_fast, 1024, [7, 7], padding='VALID', scope='fc6')
                net_fast = slim.conv2d(net_fast, 1024, [1, 1], scope='fc7')
                net_fast = tf.squeeze(net_fast, [1, 2])
                net_m_score = slim.fully_connected(net_fast, self.config.num_cls + 1, activation_fn=None,
                                                   scope='cls',
                                                   weights_initializer=tf.truncated_normal_initializer(stddev=0.01))
                net_m_t = slim.fully_connected(net_fast, (self.config.num_cls + 1) * 4, activation_fn=None,
                                               scope='box',
                                               weights_initializer=tf.truncated_normal_initializer(stddev=0.001))

        net_m_t = tf.reshape(net_m_t, (-1, self.config.num_cls + 1, 4))
        return net_m_score, net_m_t

    def mask_net(self, net_mask):
        with tf.variable_scope('mask'):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                weights_initializer=tf.variance_scaling_initializer()):
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn1')
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn2')
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn3')
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn4')

                net_mask = slim.conv2d_transpose(net_mask, 256, [2, 2], stride=2, scope='fcn5')
                net_mask = slim.conv2d(net_mask, self.config.num_cls, [1, 1], activation_fn=None, scope='mask_out', )
        net_mask = tf.transpose(net_mask, [0, 3, 1, 2])
        return net_mask

    def build_net(self, Iter):
        im, bboxes, nums, masks = Iter.get_next()
        im, bboxes, img_H, img_W = self.handle_im(im, bboxes)
        im = im - self.Mean
        im = im[..., ::-1]

        with slim.arg_scope(self.argscope):
            outputs, end_points = resnet_v1.resnet_v1_101(im, is_training=False)

        var_pre = tf.global_variables()[1:]
        C5 = end_points['resnet_v1_101/block4']
        C4 = end_points['resnet_v1_101/block3']
        C3 = end_points['resnet_v1_101/block2']
        C2 = end_points['resnet_v1_101/block1']
        C = [C5, C4, C3, C2]
        P, map_HW = self.fpn_net(C)
        rpn_net_score, rpn_net_loc = self.rpn_net(P)

        map_id = 0
        tanchors = []
        for i in range(5):
            map_H, map_W = map_HW[i]
            tanchors.append(tf.reshape(self.anchors[i][:map_H, :map_W], (-1, 4)))
        tanchors = tf.concat(tanchors, axis=0)
        inds, label, indsP, loc = self.ATC(bboxes[map_id], tanchors, (img_H, img_W))
        net_score_train = tf.gather(rpn_net_score[map_id], inds)
        net_loc_train = tf.gather(rpn_net_loc[map_id], indsP)
        rpn_cls_loss = softmaxloss(net_score_train, label)
        rpn_box_loss = SmoothL1Loss(net_loc_train, loc, 3.0, tf.to_float(tf.shape(label)[0]))
        rpn_loss = rpn_cls_loss + rpn_box_loss

        cls_loss = 0
        box_loss = 0

        roi = self.PC(rpn_net_loc[0], tf.nn.softmax(rpn_net_score)[0][:, 1], tanchors, (img_H, img_W), map_HW)
        bboxes = bboxes[0]
        masks = masks[0]

        roi, roi_inds, loc, label = self.fast_train_data_box(roi, bboxes, self.PTC_1, self.loc_std1)

        roi = tf.stop_gradient(roi)
        roi_inds = tf.stop_gradient(roi_inds)
        loc = tf.stop_gradient(loc)
        label = tf.stop_gradient(label)

        net_fast = self.pooling(P, roi, roi_inds, (img_H, img_W), map_HW, 7)
        net_m_score, net_m_loc = self.fast_net(net_fast, 'fast')
        fast_cls_loss, fast_box_loss = self.fast_loss_box(net_m_score, net_m_loc, label, loc)
        cls_loss += fast_cls_loss * self.loss_weights[0]
        box_loss += fast_box_loss * self.loss_weights[0]

        fast_loc = net_m_loc[:, 1:] * tf.constant(self.loc_std1)
        score = tf.nn.softmax(net_m_score)[:, 1:]
        inds_b = tf.argmax(score, axis=-1)
        inds_a = tf.range(tf.shape(score)[0])
        inds_b = tf.to_int32(inds_b)
        inds_a = tf.reshape(inds_a, (-1, 1))
        inds_b = tf.reshape(inds_b, (-1, 1))
        inds = tf.concat([inds_a, inds_b], axis=-1)
        fast_loc = tf.gather_nd(fast_loc, inds)

        roi = self.loc2bbox(fast_loc, roi)
        roi = self.filter_bboxes(roi, img_H, img_W, self.config.roi_min_size[0])
        roi, roi_inds, loc, label = self.fast_train_data_box(roi, bboxes, self.PTC_2, self.loc_std2)

        roi = tf.stop_gradient(roi)
        roi_inds = tf.stop_gradient(roi_inds)
        loc = tf.stop_gradient(loc)
        label = tf.stop_gradient(label)

        net_fast = self.pooling(P, roi, roi_inds, (img_H, img_W), map_HW, 7)
        net_m_score, net_m_loc = self.fast_net(net_fast, 'fast_2')
        fast_cls_loss, fast_box_loss = self.fast_loss_box(net_m_score, net_m_loc, label, loc)
        cls_loss += fast_cls_loss * self.loss_weights[1]
        box_loss += fast_box_loss * self.loss_weights[1]

        fast_loc = net_m_loc[:, 1:] * tf.constant(self.loc_std2)
        score = tf.nn.softmax(net_m_score)[:, 1:]
        inds_b = tf.argmax(score, axis=-1)
        inds_a = tf.range(tf.shape(score)[0])
        inds_b = tf.to_int32(inds_b)
        inds_a = tf.reshape(inds_a, (-1, 1))
        inds_b = tf.reshape(inds_b, (-1, 1))
        inds = tf.concat([inds_a, inds_b], axis=-1)
        fast_loc = tf.gather_nd(fast_loc, inds)

        roi = self.loc2bbox(fast_loc, roi)
        roi = self.filter_bboxes(roi, img_H, img_W, self.config.roi_min_size[0])
        roi, roi_inds, loc, label, inter, inter_inds, target = self.fast_train_data(roi, bboxes, masks, (img_H, img_W),
                                                                                    self.loc_std3)

        roi = tf.stop_gradient(roi)
        roi_inds = tf.stop_gradient(roi_inds)
        loc = tf.stop_gradient(loc)
        label = tf.stop_gradient(label)
        inter = tf.stop_gradient(inter)
        inter_inds = tf.stop_gradient(inter_inds)
        target = tf.stop_gradient(target)

        net_fast = self.pooling(P, roi, roi_inds, (img_H, img_W), map_HW, 7)
        net_mask = self.pooling(P, inter, inter_inds, (img_H, img_W), map_HW, 14)

        net_m_score, net_m_loc = self.fast_net(net_fast, 'fast_3')
        net_mask = self.mask_net(net_mask)

        fast_cls_loss, fast_box_loss = self.fast_loss_box(net_m_score, net_m_loc, label, loc)
        cls_loss += fast_cls_loss * self.loss_weights[2]
        box_loss += fast_box_loss * self.loss_weights[2]

        inds_a = tf.where(label > 0)[:, 0]
        inds_b = tf.gather(label, inds_a)
        inds_a = tf.to_int32(inds_a)
        inds_a = tf.reshape(inds_a, (-1, 1))
        inds_b = tf.reshape(inds_b, (-1, 1))

        inds = tf.concat([inds_a, inds_b - 1], axis=-1)
        net_mask_train = tf.gather_nd(net_mask, inds)
        mask_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=target, logits=net_mask_train)
        mask_loss = tf.cond(tf.equal(tf.shape(loc)[0], 0), lambda: tf.reduce_sum(mask_loss),
                            lambda: tf.reduce_mean(mask_loss))

        self.a = rpn_cls_loss
        self.b = rpn_box_loss
        self.c = cls_loss
        self.d = box_loss
        self.e = mask_loss
        self.fast_num = tf.shape(roi)[0]
        self.fast_num_P = tf.shape(loc)[0]
        loss = rpn_loss + cls_loss + box_loss + mask_loss
        return loss, var_pre

    def fast_loss_box(self, net_m_score, net_m_loc, label, loc):
        fast_num = tf.shape(label)[0]
        inds_a = tf.where(label > 0)[:, 0]
        inds_b = tf.gather(label, inds_a)
        loc = tf.gather(loc, inds_a)
        inds_a = tf.to_int32(inds_a)
        inds_a = tf.reshape(inds_a, (-1, 1))
        inds_b = tf.reshape(inds_b, (-1, 1))
        inds = tf.concat([inds_a, inds_b], axis=-1)
        net_m_loc_train = tf.gather_nd(net_m_loc, inds)
        fast_cls_loss = softmaxloss(net_m_score, label)
        fast_box_loss = SmoothL1Loss(net_m_loc_train, loc, 1.0, tf.to_float(fast_num))
        return fast_cls_loss, fast_box_loss

    def loc2bbox(self, pre_loc, anchor):
        c_hw = anchor[..., 2:4] - anchor[..., 0:2]
        c_yx = anchor[..., :2] + c_hw / 2
        yx = pre_loc[..., :2] * c_hw + c_yx
        hw = tf.exp(pre_loc[..., 2:4]) * c_hw
        yx1 = yx - hw / 2
        yx2 = yx + hw / 2
        bboxes = tf.concat((yx1, yx2), axis=-1)
        return bboxes

    def filter_bboxes(self, roi, h, w, roi_min_size):
        h = tf.to_float(h)
        w = tf.to_float(w)
        roi = tf.clip_by_value(roi, [0, 0, 0, 0], [h, w, h, w])
        hw = roi[:, 2:4] - roi[:, :2]
        inds = tf.reduce_all(hw >= roi_min_size, axis=-1)
        roi = tf.boolean_mask(roi, inds)
        return roi

    def train(self):

        shard_nums = self.config.gpus * self.config.batch_size_per_GPU
        base_lr = self.config.lr * shard_nums
        print('*****************', shard_nums, base_lr)
        steps = tf.Variable(0.0, name='steps_Faster_rcnn', trainable=False)
        x = 4
        print('x:', x)
        lr = tf.case({steps < 60000.0 * x: lambda: base_lr, steps < 80000.0 * x: lambda: base_lr / 10},
                     default=lambda: base_lr / 100)
        tower_grads = []
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        var_reuse = False
        Iter_list = []

        c = 0
        for i in range(self.config.gpus):
            with tf.device('/gpu:%d' % i):
                loss = 0
                for j in range(self.config.batch_size_per_GPU):
                    Iter = readData(self.config.files, batch_size=1, num_threads=16, shuffle_buffer=256,
                                    num_shards=shard_nums, shard_index=c)
                    Iter_list.append(Iter)
                    with tf.variable_scope('', reuse=var_reuse):
                        if c == 0:
                            pre_loss, var_pre = self.build_net(Iter)
                            var_mask = tf.global_variables('FPN') + tf.global_variables('rpn') + tf.global_variables(
                                'fast/') + tf.global_variables('mask')
                            var_fast = tf.global_variables('fast')
                            saver = tf.train.Saver(max_to_keep=200)
                        else:
                            pre_loss, _ = self.build_net(Iter)
                        var_reuse = True
                        loss += pre_loss
                    c += 1
                loss = loss / self.config.batch_size_per_GPU
                train_vars = tf.trainable_variables()
                l2_loss = tf.losses.get_regularization_losses()
                l2_re_loss = tf.add_n(l2_loss)
                faster_loss = loss + l2_re_loss
                grads_and_vars = opt.compute_gradients(faster_loss, train_vars)
                tower_grads.append(grads_and_vars)
        for v in tf.global_variables():
            print(v)
        V = tf.global_variables()
        grads = average_gradients(tower_grads)
        grads = list(zip(*grads))[0]
        grads, norm = tf.clip_by_global_norm(grads, 35.0)
        train_op = opt.apply_gradients(zip(grads, train_vars), global_step=steps)

        pre_file = '/home/zhai/PycharmProjects/Demo35/tensorflow_zoo/ResNet_caffemodel/resnet_v1_caffe/resnet_v1_101_caffe.ckpt'
        saver_pre = tf.train.Saver(var_pre)


        var_pre = var_pre + var_mask

        print(len(var_pre), len(V))
        VV = [v for v in V if v not in var_pre]
        for v in VV:
            print(v)
        print(len(VV))
        # saver_pre = tf.train.Saver(var_pre)

        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.allow_soft_placement = True

        for v in var_fast:
            print(v.name)
        with tf.Session(config=config) as sess:
            sess.run(tf.global_variables_initializer())
            saver_pre.restore(sess, pre_file)

            # saver.restore(sess, file)

            for Iter in Iter_list:
                sess.run(Iter.initializer)



            for i in range(0000, int(90010 * x)):
                if i % 20 == 0:
                    _, loss, a, b, c, d, e, f, g, h, hh, j, steps_ = sess.run(
                        [train_op, faster_loss, self.a, self.b, self.c, self.d, self.e, l2_re_loss,
                         norm, self.fast_num, self.fast_num_P, lr, steps])
                    print(datetime.now(), 'loss:%.4f' % loss, 'rpn_cls_loss:%.4f' % a, 'rpn_box_loss:%.4f' % b,
                          'fast_cls_loss:%.4f' % c, 'fast_box_loss:%.4f' % d, 'mask_loss:%.4f' % e,
                          'l2_re_loss:%.4f' % f, 'norm:%.4f' % g, h,
                          hh, j, i, steps_)
                else:
                    sess.run(train_op)

                if (i + 1) % 5000 == 0 or ((i + 1) % 1000 == 0 and i < 10000):
                    saver.save(sess, os.path.join('./models/', 'Mask_Rcnn_cascade_4x.ckpt'), global_step=i + 1)
            saver.save(sess, os.path.join('./models/', 'Mask_Rcnn_cascade_4x.ckpt'), global_step=i + 1)

            pass

    pass


import cv2


def draw_gt(im, gt):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    for box in boxes:
        # print(box)
        y1, x1, y2, x2 = box[:4]
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


if __name__ == "__main__":
    Mean = np.array([123.68, 116.78, 103.94], dtype=np.float32)
    path = '/home/zhai/PycharmProjects/Demo35/tf_cascade_rcnn/data_preprocess/'
    files = [path + 'coco_train2017.tf']
    config = Config(True, Mean, files, num_cls=80, lr=0.00125, batch_size_per_GPU=2, gpus=2,
                    roi_train_pre_nms=12000, roi_train_post_nms=2000,
                    roi_min_size=[4, 8, 16, 32, 64], fast_n_sample=512)

    Mask_Rcnn = Mask_rcnn_resnet_101(config)
    Mask_Rcnn.train()

#1x
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.364
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.556
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.397
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.193
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.400
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.485
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.309
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.486
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.510
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.295
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.553
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.656
# Loading and preparing results...
# DONE (t=2.04s)
# creating index...
# index created!
# 5000
# Running per image evaluation...
# Evaluate annotation type *segm*
# DONE (t=46.98s).
# Accumulating evaluation results...
# DONE (t=6.78s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.312
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.522
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.329
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.148
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.344
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.438
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.274
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.421
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.440
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.233
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.482
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.595

#4x
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.415
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.612
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.457
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.227
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.454
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.559
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.333
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.524
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.549
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.331
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.597
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.714
# Loading and preparing results...
# DONE (t=1.13s)
# creating index...
# index created!
# 5000
# Running per image evaluation...
# Evaluate annotation type *segm*
# DONE (t=48.95s).
# Accumulating evaluation results...
# DONE (t=6.53s).
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.352
#  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.574
#  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.373
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.177
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.385
#  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.494
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.294
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.452
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.472
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.263
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.512
#  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.644