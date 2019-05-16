# !/usr/bin/python
# -*- coding:utf-8 -*-

import os
import numpy as np

from lxml import etree
from sklearn.externals import joblib

Label = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
         'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

Label_dict = {}

c = 0
for i in Label:
    Label_dict[i] = c
    Label_dict[c] = i
    c += 1
print(Label_dict)
train_dir = r'/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/JPEGImages/'
names = os.listdir(train_dir)
names = [name.split('.')[0] for name in names]
names = sorted(names)
ann_path = '/home/zhai/PycharmProjects/Demo35/dataset/voc/VOCtest2007/VOCdevkit/VOC2007/Annotations/'
c = 0

GT = {}
for name in names:
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
        x1 = float(x1)
        x2 = float(x2)
        y1 = float(y1)
        y2 = float(y2)

        bboxes.append([x1, y1, x2, y2, difficult, Label_dict[cls], 0])
    bboxes = np.array(bboxes, dtype=np.float32)
    GT[name] = bboxes
joblib.dump(GT, 'voc_GT.pkl')
