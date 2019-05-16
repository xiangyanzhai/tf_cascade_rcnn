# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from tf_cascade_rcnn.tool.config import Config
from tf_cascade_rcnn.tool import vgg
from tf_cascade_rcnn.tool.ROIAlign import roi_align
from tf_cascade_rcnn.tool.tf_PC_test import ProposalCreator
from tf_cascade_rcnn.tool.get_anchors import get_anchors
from tf_cascade_rcnn.tool.faster_predict import predict
from sklearn.externals import joblib


class Faster_rcnn16():
    def __init__(self, config):
        self.config = config
        self.Mean = tf.constant(self.config.Mean, dtype=tf.float32)
        self.num_anchor = len(config.anchor_scales) * len(config.anchor_ratios)
        self.anchors = get_anchors(np.ceil(self.config.img_max / 16), self.config.anchor_scales,
                                   self.config.anchor_ratios)

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms, n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

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

    def rpn_net(self, net):
        with tf.variable_scope('rpn'):
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay),
                                weights_initializer=tf.truncated_normal_initializer(stddev=0.01)):
                net_rpn = slim.conv2d(net, 512, [3, 3], scope='conv')
                net_score = slim.conv2d(net_rpn, self.num_anchor * 2, [1, 1], activation_fn=None, scope='cls')
                net_t = slim.conv2d(net_rpn, self.num_anchor * 4, [1, 1], activation_fn=None, scope='box')
        m = tf.shape(net)[0]
        net_score = tf.reshape(net_score, [m, -1, 2])
        net_t = tf.reshape(net_t, [m, -1, 4])
        return net_score, net_t

    def roi_layer(self, loc, score, anchor, img_size):
        roi = self.PC(loc, score, anchor, img_size, train=self.config.is_train)
        roi_inds = tf.zeros(tf.shape(roi)[0], dtype=tf.int32)

        return roi, roi_inds

    def pooling(self, net, roi, roi_inds, img_H, img_W, map_H, map_W):
        img_H = tf.to_float(img_H)
        img_W = tf.to_float(img_W)

        map_H = tf.to_float(map_H)
        map_W = tf.to_float(map_W)
        roi_norm = roi / tf.concat([[img_H], [img_W], [img_H], [img_W]], axis=0) * tf.concat(
            [[map_H], [map_W], [map_H], [map_W]], axis=0)
        roi_norm = tf.stop_gradient(roi_norm)
        net_fast = roi_align(net, roi_norm, roi_inds, 7)
        return net_fast

    def fast_net(self, net_fast):
        with tf.variable_scope('vgg_16', reuse=True):
            net_fast = slim.conv2d(net_fast, 4096, [7, 7], padding='VALID', scope='fc6')

            net_fast = slim.dropout(net_fast, 0.5, is_training=False,
                                    scope='dropout6')
            net_fast = slim.conv2d(net_fast, 4096, [1, 1], scope='fc7')
            net_fast = slim.dropout(net_fast, 0.5, is_training=False,
                                    scope='dropout7')
            net_fast = tf.squeeze(net_fast, [1, 2])
        with tf.variable_scope('fast') as scope:
            with slim.arg_scope([slim.conv2d, slim.fully_connected],
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
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

        with slim.arg_scope(vgg.vgg_arg_scope()):
            outputs, end_points = vgg.vgg_16(im, num_classes=None)
        net = end_points['vgg_16/conv5/conv5_3']
        map_H = tf.shape(net)[1]
        map_W = tf.shape(net)[2]
        rpn_net_score, rpn_net_loc = self.rpn_net(net)

        tanchors = self.anchors[:map_H, :map_W]
        tanchors = tf.reshape(tanchors, (-1, 4))

        roi, roi_inds = self.roi_layer(rpn_net_loc[0], tf.nn.softmax(rpn_net_score)[0][:, 1], tanchors, (img_H, img_W))
        net_fast = self.pooling(net, roi, roi_inds, img_H, img_W, map_H, map_W)
        net_m_score, net_m_t = self.fast_net(net_fast)
        net_m_t = tf.reshape(net_m_t, (-1, self.config.num_cls + 1, 4)) * tf.constant([0.1, 0.1, 0.2, 0.2])

        self.result = predict(net_m_t, net_m_score, roi, img_H, img_W)

    def test(self):
        self.build_net()
        file = '../train/models/Faster_Rcnn_vgg16.ckpt-90000'
        # file = '/home/zhai/PycharmProjects/Demo35/my_Faster_tool/train/models/Faster_vgg16_ohem.ckpt-89999'
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
            time_start = datetime.now()
            for name in names[:m]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                res, s = sess.run([self.result, self.scale], feed_dict={self.im_input: img})
                res[:, :4] = res[:, :4] / s
                res = res[:, [1, 0, 3, 2, 4, 5]]
                Res[name] = res
            print(datetime.now() - time_start)
            joblib.dump(Res, 'Faster_Rcnn_vgg16.pkl')
            GT = joblib.load('../mAP/voc_GT.pkl')
            AP = mAP(Res, GT, 20, iou_thresh=0.5, use_07_metric=True, e=0.01)
            print(AP)
            AP = AP.mean()
            print(AP)

        pass

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
    config = Config(False, Mean, None, lr=0.001, weight_decay=0.0005, num_cls=20, img_max=1000, img_min=600,
                    anchor_scales=[128, 256, 512], anchor_ratios=[0.5, 1, 2],
                    roi_train_pre_nms=12000, roi_train_post_nms=2000,
                    roi_test_pre_nms=6000, roi_test_post_nms=300, roi_min_size=16,
                    fast_n_sample=128)

    faster_rcnn = Faster_rcnn16(config)
    faster_rcnn.test()
