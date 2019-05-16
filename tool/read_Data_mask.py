# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as  tf
import numpy as np
import cv2


def bbox_flip_left_right(bboxes, w):
    w = tf.to_float(w)
    y1, x1, y2, x2, cls = tf.split(bboxes, 5, axis=1)
    x1, x2 = w - 1. - x2, w - 1. - x1
    return tf.concat([y1, x1, y2, x2, cls], axis=1)


# def decode_mask(x):
#
#     return maskUtils(x)
def py_decode_mask(counts, mask_h, mask_w):
    if len(counts)==0:
        mask=np.zeros((0,mask_h,mask_w),dtype=np.uint8)
        return mask
    mask = map(lambda x: maskUtils.decode({'size': [mask_h, mask_w], 'counts': x}), counts)
    mask = list(mask)
    mask = np.array(mask)
    mask = mask.astype(np.uint8)
    return mask


def parse(se):
    features = tf.parse_single_example(
        se, features={
            'im': tf.FixedLenFeature([], tf.string),
            'bboxes': tf.FixedLenFeature([], tf.string),
            'Counts': tf.VarLenFeature(tf.string),
            'Id': tf.FixedLenFeature([], tf.string),
        }
    )
    img = features['im']
    bboxes = features['bboxes']
    Counts = features['Counts']
    Id = features['Id']
    img = tf.image.decode_jpeg(img, channels=3)
    bboxes = tf.decode_raw(bboxes, tf.float32)

    bboxes = tf.reshape(bboxes, (-1, 6))
    inds = bboxes[:, -1]
    bboxes = bboxes[:, :-1]

    Counts = tf.sparse_tensor_to_dense(Counts, default_value=b'')
    mask_h, mask_w = tf.shape(img)[0], tf.shape(img)[1]
    mask = tf.py_func(py_decode_mask, [Counts, mask_h, mask_w], tf.uint8)
    img, bboxes, mask = tf.cond(tf.random_uniform(shape=()) > 0.5, lambda: (
        tf.image.flip_left_right(img), bbox_flip_left_right(bboxes, tf.shape(img)[1]), mask[..., ::-1]),
                                lambda: (img, bboxes, mask))
    return img, bboxes, tf.shape(bboxes)[0], mask


def readData(files, batch_size=1, num_epochs=2000, num_threads=16, shuffle_buffer=1024, num_shards=1, shard_index=0):
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse, num_parallel_calls=num_threads)

    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None, 3], [None, 5], [], [None, None, None],),
                                   padding_values=(np.array(127, dtype=np.uint8), np.array(0.0, dtype=np.float32),
                                                   np.array(0.0, dtype=np.int32), np.array(0.0, dtype=np.uint8)))
    # dataset = dataset.batch(batch_size)

    iter = dataset.make_initializable_iterator()
    return iter


def draw_gt(im, gt):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    for box in boxes:
        # print(box)
        y1, x1, y2, x2 = box[:4]

        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
        # print(box[-1])
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


from pycocotools import mask as maskUtils


def draw_mask(im, gt, mask):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)

    c = 0
    for box in boxes:
        # print(box)

        t = mask[c, :, :]

        inds = np.where(t == 1)
        im[inds] = np.array([0, 0, 255], )
        y1, x1, y2, x2 = box[:4]

        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
        print(box[-1])
        c += 1
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


import os
from datetime import datetime

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    from sklearn.externals import joblib

    # files = ['./data/voc2007.tf', './data/voc2012.tf']

    # path = '/home/zhai/PycharmProjects/Demo35/dataset/'
    # files = [path]
    # # files = ['./data/voc2007.tf']
    #
    # files = [path + 'voc2007.tf', path + 'voc2012.tf']
    # files = [path + 'voc2007.tf']
    path = '/home/zhai/PycharmProjects/Demo35/data_set_yxyx/'
    files = [path + 'coco_train2017.tf']

    path='/home/zhai/PycharmProjects/Demo35/tf_cascade_rcnn/data_preprocess/'
    files=[path+'res_round2.tf']
    files = [path + 'norm_round2.tf']
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    Iter = readData(files, shuffle_buffer=None)
    # Iter1 = readData(files)
    # Iter2 = readData(files)
    im, bboxes, nums, mask = Iter.get_next()

    with tf.Session(config=config) as sess:

        sess.run(Iter.initializer)
        for i in range(1000000):
            img, bbox, num_, mask_ = sess.run([im, bboxes, nums, mask])
            # tt = mask_[0]
            # print(tt.shape, num_[0], bbox[0].shape)
            print(bbox.shape,mask_.shape)
            draw_gt(img[0], bbox[0][:])
            #
            draw_mask(img[0], bbox[0][:], mask_[0][:])
            # #
            # print(datetime.now(),i,mask.shape)
            # print(e,f,g,h,h[0]==False)
            print(datetime.now(), i)
        #
        # draw_gt(a,b)
    # joblib.dump((a,b),'(t_im,t_bboxes).pkl')
