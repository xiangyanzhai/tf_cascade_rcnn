# !/usr/bin/python
# -*- coding:utf-8 -*-
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from datetime import datetime
from tf_cascade_rcnn.tool.ROIAlign import roi_align
from tf_cascade_rcnn.tool.tf_PC_FPN import ProposalCreator
from tf_cascade_rcnn.tool.get_anchors import get_anchors
from tf_cascade_rcnn.tool import resnet_v1
from tf_cascade_rcnn.tool.cascade_predict import predict
from tf_cascade_rcnn.tool.config import Config
from sklearn.externals import joblib
from pycocotools import mask as maskUtils
import codecs
import json
import eval_coco_box
import eval_coco_segm


# from tf_cascade_rcnn.tool.RoIAlign_NHWC import roi_align
# from tf_cascade_rcnn.tool.tf_PC_FPN_cpu import ProposalCreator


# rpn_var=np.array([1,1,1,1],dtype=np.float32)
def loadNumpyAnnotations(data):
    """
    Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
    :param  data (numpy.ndarray)
    :return: annotations (python nested list)
    """
    print('Converting ndarray to lists...')
    assert (type(data) == np.ndarray)
    print(data.shape)
    assert (data.shape[1] == 7)
    N = data.shape[0]
    ann = []
    for i in range(N):
        if i % 1000000 == 0:
            print('{}/{}'.format(i, N))

        ann += [{
            'image_id': int(data[i, 0]),
            'bbox': [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
            'score': data[i, 5],
            'category_id': int(data[i, 6]),
        }]

    return ann


def loadNumpyAnnotations_mask(data, mask):
    global oh, ow

    t = {
        'image_id': int(data[0]),
        'bbox': [data[1], data[2], data[3], data[4]],
        'score': data[5],
        'category_id': int(data[6]),
    }

    res_mask = np.zeros((oh, ow), dtype=np.uint8, order='F')
    bbox = t['bbox']
    bbox = np.round(bbox)
    bbox = bbox.astype(np.int32)
    x1 = bbox[0]
    y1 = bbox[1]
    w = bbox[2]
    h = bbox[3]
    x2 = x1 + w
    y2 = y1 + h

    x1 = np.clip(x1, 0, ow)
    x2 = np.clip(x2, 0, ow)

    y1 = np.clip(y1, 0, oh)
    y2 = np.clip(y2, 0, oh)
    w = x2 - x1
    h = y2 - y1

    img = cv2.resize(mask, (w, h))
    img = np.round(img)
    img = img.astype(np.uint8)

    res_mask[y1:y2, x1:x2] = img
    tt = maskUtils.encode(res_mask)
    tt['counts'] = tt['counts'].decode('utf-8')
    t["segmentation"] = tt

    return t


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

        self.PC = ProposalCreator(nms_thresh=config.roi_nms_thresh,
                                  n_train_pre_nms=config.roi_train_pre_nms, n_train_post_nms=config.roi_train_post_nms,
                                  n_test_pre_nms=config.roi_test_pre_nms, n_test_post_nms=config.roi_test_post_nms,
                                  min_size=config.roi_min_size)

        self.argscope = resnet_v1.resnet_arg_scope(weight_decay=config.weight_decay)
        self.loc_std1 = [1. / 10, 1. / 10, 1. / 5, 1. / 5]
        self.loc_std2 = [1. / 20, 1. / 20, 1. / 10, 1. / 10]
        self.loc_std3 = [1. / 30, 1. / 30, 1. / 15, 1. / 15]
        self.weights = [1.0, 1.0, 1.0]

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

    def roi_layer(self, roi):
        area = tf.reduce_prod(roi[:, 2:4] - roi[:, :2], axis=1)
        roi_inds = tf.floor(4.0 + tf.log(area ** 0.5 / 224.0) / tf.log(2.0))
        roi_inds = tf.clip_by_value(roi_inds, 2, 5)
        roi_inds = roi_inds - 2
        roi_inds = tf.to_int32(roi_inds)
        return roi, roi_inds

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
                                weights_regularizer=slim.l2_regularizer(self.config.weight_decay)):
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn1')
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn2')
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn3')
                net_mask = slim.conv2d(net_mask, 256, [3, 3], scope='fcn4')

                net_mask = slim.conv2d_transpose(net_mask, 256, [2, 2], stride=2, scope='fcn5')
                net_mask = slim.conv2d(net_mask, self.config.num_cls, [1, 1], activation_fn=None, scope='mask_out', )
        net_mask = tf.transpose(net_mask, [0, 3, 1, 2])
        return net_mask

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
        for c in C:
            print(c)
        P, map_HW = self.fpn_net(C)
        rpn_net_score, rpn_net_loc = self.rpn_net(P)

        tanchors = []
        for i in range(5):
            map_H, map_W = map_HW[i]
            tanchors.append(tf.reshape(self.anchors[i][:map_H, :map_W], (-1, 4)))
        tanchors = tf.concat(tanchors, axis=0)

        roi = self.PC(rpn_net_loc[0], tf.nn.softmax(rpn_net_score)[0][:, 1],
                      tanchors, (img_H, img_W), map_HW, train=self.config.is_train)

        roi, roi_inds = self.roi_layer(roi)
        net_fast = self.pooling(P, roi, roi_inds, (img_H, img_W), map_HW, 7)
        net_m_score, net_m_loc = self.fast_net(net_fast, 'fast')

        fast_loc = net_m_loc[:, 1:] * tf.constant(self.loc_std1)
        roi = self.loc2bbox(fast_loc, roi[:, None])
        score = tf.nn.softmax(net_m_score)[:, 1:]
        pre_bboxes = roi * self.weights[0]
        pre_score = score * self.weights[0]

        inds_b = tf.argmax(score, axis=-1)
        inds_a = tf.range(tf.shape(score)[0])
        inds_b = tf.to_int32(inds_b)
        inds_a = tf.reshape(inds_a, (-1, 1))
        inds_b = tf.reshape(inds_b, (-1, 1))
        inds = tf.concat([inds_a, inds_b], axis=-1)
        roi = tf.gather_nd(roi, inds)

        roi, inds = self.filter_bboxes(roi, img_H, img_W, self.config.roi_min_size[0])
        pre_bboxes = tf.boolean_mask(pre_bboxes, inds)
        pre_score = tf.boolean_mask(pre_score, inds)
        roi, roi_inds = self.roi_layer(roi)
        net_fast = self.pooling(P, roi, roi_inds, (img_H, img_W), map_HW, 7)
        net_m_score, net_m_loc = self.fast_net(net_fast, 'fast_2')

        fast_loc = net_m_loc[:, 1:] * tf.constant(self.loc_std2)
        roi = self.loc2bbox(fast_loc, roi[:, None])
        score = tf.nn.softmax(net_m_score)[:, 1:]
        pre_bboxes += roi * self.weights[1]
        pre_score += score * self.weights[1]

        inds_b = tf.argmax(score, axis=-1)
        inds_a = tf.range(tf.shape(score)[0])
        inds_b = tf.to_int32(inds_b)
        inds_a = tf.reshape(inds_a, (-1, 1))
        inds_b = tf.reshape(inds_b, (-1, 1))
        inds = tf.concat([inds_a, inds_b], axis=-1)
        roi = tf.gather_nd(roi, inds)

        roi, inds = self.filter_bboxes(roi, img_H, img_W, self.config.roi_min_size[0])
        pre_bboxes = tf.boolean_mask(pre_bboxes, inds)
        pre_score = tf.boolean_mask(pre_score, inds)
        roi, roi_inds = self.roi_layer(roi)
        net_fast = self.pooling(P, roi, roi_inds, (img_H, img_W), map_HW, 7)
        net_m_score, net_m_loc = self.fast_net(net_fast, 'fast_3')

        fast_loc = net_m_loc[:, 1:] * tf.constant(self.loc_std3)
        roi = self.loc2bbox(fast_loc, roi[:, None])
        score = tf.nn.softmax(net_m_score)[:, 1:]
        pre_bboxes += roi * self.weights[2]
        pre_score += score * self.weights[2]

        pre_bboxes = pre_bboxes / sum(self.weights)
        pre_score = pre_score / sum(self.weights)

        self.result = predict(pre_bboxes, pre_score, img_H, img_W, iou_thresh_=0.5, c_thresh=0.05)[:100]

        roi = self.result[:100]
        area = tf.reduce_prod(roi[:, 2:4] - roi[:, :2] + 1, axis=1)
        roi_inds = tf.floor(4.0 + tf.log(area ** 0.5 / 224.0) / tf.log(2.0))
        roi_inds = tf.clip_by_value(roi_inds, 2, 5)
        roi_inds = roi_inds - 2
        roi_inds = tf.to_int32(roi_inds)

        net_mask = self.pooling(P, roi[:, :4], roi_inds, (img_H, img_W), map_HW, 14)
        net_mask = self.mask_net(net_mask)
        inds_a = tf.range(tf.shape(roi)[0])
        inds_b = tf.to_int32(roi[:, -1])
        inds = tf.concat([tf.reshape(inds_a, (-1, 1)), tf.reshape(inds_b, (-1, 1))], axis=-1)

        mask = tf.gather_nd(net_mask, inds)
        self.mask = tf.nn.sigmoid(mask)

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
        return roi, inds

    def test(self):
        global oh, ow
        self.build_net()
        for v in tf.global_variables():
            print(v)
        catId2cls, cls2catId, catId2name = joblib.load(
            '(catId2cls,cls2catId,catId2name).pkl')

        file = '/home/zhai/PycharmProjects/Demo35/tf_cascade_rcnn/train/models/Mask_Rcnn_cascade_4x.ckpt-360000'
        saver = tf.train.Saver()

        test_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/coco/val2017/'
        names = os.listdir(test_dir)
        names = [name.split('.')[0] for name in names]
        names = sorted(names)

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with tf.Session(config=config) as sess:
            print(file)
            saver.restore(sess, file)

            i = 0
            mm = 100000000
            Res = []
            Res_mask = []
            time_start = datetime.now()
            for name in names[:mm]:
                i += 1
                print(datetime.now(), i)
                im_file = test_dir + name + '.jpg'
                img = tf.gfile.FastGFile(im_file, 'rb').read()
                cv_img = cv2.imread(im_file)
                oh, ow = cv_img.shape[:2]
                res, s, res_mask = sess.run([self.result, self.scale, self.mask], feed_dict={self.im_input: img})
                res[:, :4] = res[:, :4] / s
                res = res[:, [1, 0, 3, 2, 4, 5]]
                wh = res[:, 2:4] - res[:, :2] + 1

                imgId = int(name)
                m = res.shape[0]

                imgIds = np.zeros((m, 1)) + imgId

                cls = res[:, 5]
                cid = map(lambda x: cls2catId[x], cls)
                cid = list(cid)
                cid = np.array(cid)
                cid = cid.reshape(-1, 1)

                res = np.concatenate((imgIds, res[:, :2], wh, res[:, 4:5], cid), axis=1)
                # Res=np.concatenate([Res,res])
                res = np.round(res, 4)
                Res.append(res)
                Res_mask += map(loadNumpyAnnotations_mask, res[:100], res_mask[:100])

                # Res[name] = res
            Res = np.concatenate(Res, axis=0)

            Ann = loadNumpyAnnotations(Res)
            print('==================================', mm, datetime.now() - time_start)

            with codecs.open('Mask_Rcnn_bbox.json', 'w', 'ascii') as f:
                json.dump(Ann, f)
            with codecs.open('Mask_Rcnn_segm.json', 'w', 'ascii') as f:
                json.dump(Res_mask, f)
            eval_coco_box.eval('Mask_Rcnn_bbox.json', mm)
            eval_coco_segm.eval('Mask_Rcnn_segm.json', mm)


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

    config = Config(False, Mean, None, roi_min_size=[4, 8, 16, 32, 64], roi_test_pre_nms=6000, roi_test_post_nms=1000)

    Mask_rcnn = Mask_rcnn_resnet_101(config)
    Mask_rcnn.test()
