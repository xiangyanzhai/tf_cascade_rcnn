# !/usr/bin/python
# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import pycocotools
from sklearn.externals import joblib


pylab.rcParams['figure.figsize'] = (10.0, 8.0)
import tensorflow as tf
import cv2

annType = ['segm', 'bbox', 'keypoints']
annType = annType[0]  # specify type here
prefix = 'person_keypoints' if annType == 'keypoints' else 'instances'
print('Running demo for *%s* results.' % (annType))

# dataType = 'val2017'
# dataDir = '/home/zhai/PycharmProjects/Demo35/dataset/coco'
# annFile = '%s/annotations/%s_%s.json' % (dataDir, prefix, dataType)
annFile='/home/zhai/PycharmProjects/Demo35/dataset/coco/instances_minival2014.json'
annFile='/home/zhai/PycharmProjects/Demo35/dataset/coco/annotations/instances_val2017.json'
cocoGt = COCO(annFile)
imgIds=cocoGt.getImgIds()
imgIds=sorted(imgIds)
imgIds=imgIds[:]
# resFile='/home/zhai/PycharmProjects/Demo35/myDNN/train_coco/bbox_coco_2014_minival_results_fpn.json'
# resFile = './%s_res.json'%dataType
# resFile='val2017_res_50_roi.json'

def eval(resFile,m):
    cocoDt=cocoGt.loadRes(resFile)

    cocoEval = COCOeval(cocoGt,cocoDt,annType)


    print(len(imgIds))
    cocoEval.params.imgIds  =imgIds[:m]

    cocoEval.evaluate()

    cocoEval.accumulate()

    cocoEval.summarize()
#
if __name__ == "__main__":
    import codecs
    import json




    m=5000
    e=0.05
    resFile='py_Mask_Rcnn_segm.json'
    with codecs.open(resFile, 'rb', 'ascii') as f:
        res=json.load(f)
    res=[r for r in res if r['score']>=e]
    with codecs.open('Mask_Rcnn_segm_e.json', 'w', 'ascii') as f:
        json.dump(res, f)
    resFile='Mask_Rcnn_segm_e.json'
    eval(resFile,m)
    pass
