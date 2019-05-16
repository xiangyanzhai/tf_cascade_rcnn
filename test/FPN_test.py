# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from tf_cascade_rcnn.tool.config import Config
from tf_cascade_rcnn.tool.ROIAlign import roi_align
from tf_cascade_rcnn.tool.tf_PC_FPN import ProposalCreator
from tf_cascade_rcnn.tool.get_anchors import get_anchors
from tf_cascade_rcnn.tool import resnet_v1
from tf_cascade_rcnn.tool.faster_predict import predict

from sklearn.externals import joblib

class Faster_rcnn_resnet_101_FPN():
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

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms, n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

        self.argscope = resnet_v1.resnet_arg_scope(weight_decay=config.weight_decay)

    def handle_im(self, im):
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

        return im, nh, nw, scale

    def fpn_net(self, C):
        C5, C4, C3, C2 = C
        with tf.variable_scope('FPN'):
            with slim.arg_scope([slim.conv2d, ], weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
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
                net_t2 = slim.conv2d(rpn_P2, a * 4, [1, 1], activation_fn=None, scope='box')
                scope.reuse_variables()

                rpn_P3 = slim.conv2d(P3, channel, [3, 3], scope='conv')
                rpn_P4 = slim.conv2d(P4, channel, [3, 3], scope='conv')
                rpn_P5 = slim.conv2d(P5, channel, [3, 3], scope='conv')
                rpn_P6 = slim.conv2d(P6, channel, [3, 3], scope='conv')

                net_score3 = slim.conv2d(rpn_P3, b * 2, [1, 1], activation_fn=None, scope='cls')
                net_t3 = slim.conv2d(rpn_P3, b * 4, [1, 1], activation_fn=None, scope='box')

                net_score4 = slim.conv2d(rpn_P4, c * 2, [1, 1], activation_fn=None, scope='cls')
                net_t4 = slim.conv2d(rpn_P4, c * 4, [1, 1], activation_fn=None, scope='box')

                net_score5 = slim.conv2d(rpn_P5, d * 2, [1, 1], activation_fn=None, scope='cls')
                net_t5 = slim.conv2d(rpn_P5, d * 4, [1, 1], activation_fn=None, scope='box')

                net_score6 = slim.conv2d(rpn_P6, e * 2, [1, 1], activation_fn=None, scope='cls')
                net_t6 = slim.conv2d(rpn_P6, e * 4, [1, 1], activation_fn=None, scope='box')
        m = tf.shape(P2)[0]
        net_score2 = tf.reshape(net_score2, (m, -1, 2))
        net_t2 = tf.reshape(net_t2, (m, -1, 4))

        net_score3 = tf.reshape(net_score3, (m, -1, 2))
        net_t3 = tf.reshape(net_t3, (m, -1, 4))

        net_score4 = tf.reshape(net_score4, (m, -1, 2))
        net_t4 = tf.reshape(net_t4, (m, -1, 4))

        net_score5 = tf.reshape(net_score5, (m, -1, 2))
        net_t5 = tf.reshape(net_t5, (m, -1, 4))

        net_score6 = tf.reshape(net_score6, (m, -1, 2))
        net_t6 = tf.reshape(net_t6, (m, -1, 4))

        net_score = tf.concat([net_score2, net_score3, net_score4, net_score5, net_score6], axis=1)
        net_t = tf.concat([net_t2, net_t3, net_t4, net_t5, net_t6], axis=1)
        return net_score, net_t

    def roi_layer(self, loc, score, anchor, img_size, map_HW):
        roi = self.PC(loc, score, anchor, img_size, map_HW, train=self.config.is_train)

        roi.set_shape(tf.TensorShape([None, 4]))
        area = tf.reduce_prod(roi[:, 2:4] - roi[:, :2], axis=1)
        roi_inds = tf.floor(4.0 + tf.log(area ** 0.5 / 224.0) / tf.log(2.0))
        roi_inds = tf.clip_by_value(roi_inds, 2, 5)
        roi_inds = roi_inds - 2
        roi_inds = tf.to_int32(roi_inds)
        return roi, roi_inds

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

    def fast_net(self, net_fast):
        with tf.variable_scope('fast'):
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

    def build_net(self):
        self.im_input = tf.placeholder(tf.string, name='input')
        im = tf.image.decode_jpeg(self.im_input, 3)
        im = im[None]
        im, img_H, img_W, self.scale = self.handle_im(im)
        im = im - self.Mean
        im = im[..., ::-1]

        with slim.arg_scope(self.argscope):
            outputs, end_points = resnet_v1.resnet_v1_101(im, is_training=False)

        C5 = end_points['resnet_v1_101/block4']
        C4 = end_points['resnet_v1_101/block3']
        C3 = end_points['resnet_v1_101/block2']
        C2 = end_points['resnet_v1_101/block1']
        C = [C5, C4, C3, C2]
        P, map_HW = self.fpn_net(C)
        rpn_net_score, rpn_net_loc = self.rpn_net(P)

        tanchors = []
        for i in range(5):
            map_H, map_W = map_HW[i]
            tanchors.append(tf.reshape(self.anchors[i][:map_H, :map_W], (-1, 4)))
        tanchors = tf.concat(tanchors, axis=0)

        roi, roi_inds = self.roi_layer(rpn_net_loc[0], tf.nn.softmax(rpn_net_score)[0][:, 1],
                                       tanchors, (img_H, img_W), map_HW)
        roi = tf.stop_gradient(roi)
        roi_inds = tf.stop_gradient(roi_inds)

        net_fast = self.pooling(P, roi, roi_inds, (img_H, img_W), map_HW, 7)
        net_m_score, net_m_t = self.fast_net(net_fast)

        net_m_t = tf.reshape(net_m_t, (-1, self.config.num_cls + 1, 4)) * tf.constant([0.1, 0.1, 0.2, 0.2])

        self.result = predict(net_m_t, net_m_score, roi, img_H, img_W)

    def test(self):
        self.build_net()

        file = '../train/models/FPN_101.ckpt-90000'
        # file='/home/zhai/PycharmProjects/Demo35/object_detection/train/models/FPN_101.ckpt-90000'
        saver = tf.train.Saver()
        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            saver.restore(sess, file)
            Res = {}
            i = 0
            m = 100
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res, s = sess.run([self.result, self.scale], feed_dict={self.im_input: img})
                res[:, :4] = res[:, :4] / s
                res = res[:, [1, 0, 3, 2, 4, 5]]

                Res[name] = res

        joblib.dump(Res, 'FPN_101.pkl')
        joblib.dump(Res, 'Faster_Rcnn_resnet101.pkl')
        GT = joblib.load('../mAP/voc_GT.pkl')
        AP = mAP(Res, GT, 20, iou_thresh=0.5, use_07_metric=True, e=0.01)
        print(AP)
        AP = AP.mean()
        print(AP)

    pass

from tf_cascade_rcnn.mAP.voc_mAP import mAP
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

    config = Config(False, Mean, None, lr=0.00125,num_cls=20)

    faster_rcnn = Faster_rcnn_resnet_101_FPN(config)
    faster_rcnn.test()
