# !/usr/bin/python
# -*- coding:utf-8 -*-
import os
import numpy as np
from lxml import etree
import tensorflow as  tf


def _float(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def getexample(im, bboxes, Id):
    return tf.train.Example(features=tf.train.Features(feature={
        'im': _bytes(im),
        'bboxes': _bytes(bboxes),
        'Id': _bytes(Id),
    }))


Label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

Label_dict = {}

c = 0
for i in Label:
    Label_dict[i] = c
    Label_dict[c] = i
    c += 1
print(Label_dict)


def preprocess(ann_path, image_path, tf_name):
    writer = tf.python_io.TFRecordWriter(tf_name)
    names = os.listdir(ann_path)
    names = [name.split('.')[0] for name in names]
    names = sorted(names)
    print(len(names))
    c = 0
    np.random.seed(50)
    np.random.shuffle(names)
    print('shuffle')
    for name in names[:]:
        c += 1
        print(c)
        htm = etree.parse(ann_path + name + '.xml')
        bboxes = []
        for obj in htm.xpath('//object'):
            cls = obj.xpath('name')[0].xpath('string(.)').strip()
            difficult = obj.xpath('difficult')[0].xpath('string(.)').strip()
            bbox = obj.xpath('bndbox')[0]
            x1 = bbox.xpath('xmin')[0].xpath('string(.)').strip()
            x2 = bbox.xpath('xmax')[0].xpath('string(.)').strip()
            y1 = bbox.xpath('ymin')[0].xpath('string(.)').strip()
            y2 = bbox.xpath('ymax')[0].xpath('string(.)').strip()
            difficult = int(difficult)
            x1 = float(x1)-1
            x2 = float(x2)-1
            y1 = float(y1)-1
            y2 = float(y2)-1

            if difficult == 0:
                bboxes.append([y1, x1, y2, x2, Label_dict[cls]])
        if len(bboxes) == 0:
            continue
        bboxes = np.array(bboxes)
        bboxes = bboxes.astype(np.float32)

        file = image_path + name + '.jpg'
        im = tf.gfile.FastGFile(file, 'rb').read()
        example = getexample(im, bboxes.tobytes(), name.encode('utf-8'))
        writer.write(example.SerializeToString())
    writer.close()


if __name__ == "__main__":
    ann_voc_07 = '/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCdevkit/VOC2007/Annotations/'
    image_voc_07='/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCdevkit/VOC2007/JPEGImages/'
    preprocess(ann_voc_07,image_voc_07,'voc_07.tf')
    ann_voc_12 = '/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCdevkit/VOC2012/Annotations/'
    image_voc_12 = '/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCdevkit/VOC2012/JPEGImages/'
    preprocess(ann_voc_12, image_voc_12, 'voc_12.tf')
    pass
