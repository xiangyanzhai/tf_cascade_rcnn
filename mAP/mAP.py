# !/usr/bin/python
# -*- coding:utf-8 -*-


# GT: xyxy cls  flag  m*6
# pre_bboxes: xyxy score cls

import numpy as np


def cal_IOU(pre_bboxes, bboxes):
    hw = pre_bboxes[:, 2:4] - pre_bboxes[:, :2] + 1
    areas1 = hw.prod(axis=-1)
    hw = bboxes[:, 2:4] - bboxes[:, :2] + 1
    areas2 = hw.prod(axis=-1)

    yx1 = np.maximum(pre_bboxes[:, None, :2], bboxes[:, :2])
    yx2 = np.minimum(pre_bboxes[:, None, 2:4], bboxes[:, 2:4])

    hw = yx2 - yx1 + 1
    hw = np.maximum(hw, 0)
    areas_i = hw.prod(axis=-1)
    iou = areas_i / (areas1[:, None] + areas2 - areas_i)
    return iou


# def cal_IOU(bb,BBGT):
#     ixmin = np.maximum(BBGT[:, 0], bb[0])
#     iymin = np.maximum(BBGT[:, 1], bb[1])
#     ixmax = np.minimum(BBGT[:, 2], bb[2])
#     iymax = np.minimum(BBGT[:, 3], bb[3])
#     iw = np.maximum(ixmax - ixmin + 1., 0.)
#     ih = np.maximum(iymax - iymin + 1., 0.)
#     inters = iw * ih
#
#     # union
#     uni = ((bb[2] - bb[0] + 1.) * (bb[3] - bb[1] + 1.) +
#            (BBGT[:, 2] - BBGT[:, 0] + 1.) *
#            (BBGT[:, 3] - BBGT[:, 1] + 1.) - inters)
#
#     overlaps = inters / uni
#     return overlaps
def voc_ap(rec, prec, use_07_metric=False):
    """ ap = voc_ap(rec, prec, [use_07_metric])
    Compute VOC AP given precision and recall.
    If use_07_metric is true, uses the
    VOC 07 11 point method (default:False).
    """
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def eval_ap(pre_bboxes, GT, class_id, iou_thresh=0.5, use_07_metric=True):
    names = GT.keys()
    names = sorted(names)
    name_to_id = {}
    npos = 0
    for i in range(len(names)):
        name_to_id[i] = names[i]
        name_to_id[names[i]] = i
        gt = GT[names[i]]

        inds = gt[:, -2] == class_id

        npos += inds.sum()
    pre = []
    names = pre_bboxes.keys()
    names = sorted(names)

    for name in names:
        bboxes = pre_bboxes[name]
        inds = bboxes[:, -1] == class_id
        bboxes = bboxes[inds]
        bboxes[:, -1] = name_to_id[name]
        pre.append(bboxes)
    pre = np.concatenate(pre, axis=0)

    inds = pre[:, -2].argsort()[::-1]
    pre = pre[inds]

    m = pre.shape[0]
    tp = np.zeros(m)
    fp = np.zeros(m)
    i = 0
    for bbox in pre:
        name = name_to_id[bbox[-1]]
        gt = GT[name]
        index = np.arange(gt.shape[0])
        inds = gt[:, -2] == class_id
        gt_class_bboxes = gt[inds]
        index = index[inds]

        if gt_class_bboxes.shape[0] == 0:
            fp[i] = 1
        else:
            IOU = cal_IOU(bbox[None], gt_class_bboxes)
            IOU = IOU.ravel()
            if IOU.max() <= iou_thresh:
                fp[i] = 1
            else:
                index = index[IOU.argmax()]

                if gt[index][-1] == 0:
                    tp[i] = 1
                    gt[index][-1] = 1
                else:
                    fp[i] = 1

        i += 1
    # print(fp.sum(), tp.sum(), npos)
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = voc_ap(rec, prec, use_07_metric)

    return ap



from sklearn.externals import joblib


def mAP(pre_bboxes, GT, num_cls, iou_thresh=0.5, use_07_metric=True, e=0.01):
    for name in pre_bboxes.keys():
        bboxes = pre_bboxes[name]
        inds = bboxes[:, -2] >= e
        bboxes = bboxes[inds]
        pre_bboxes[name] = bboxes
    AP = np.zeros(num_cls)

    for i in range(num_cls):
        AP[i] = eval_ap(pre_bboxes, GT, i, iou_thresh=iou_thresh, use_07_metric=use_07_metric)

    return AP




if __name__ == "__main__":
    GT = joblib.load('voc_GT.pkl')
    for name in GT.keys():
        gt = GT[name]
        inds = gt[:, 4] == 0
        gt = gt[:, [0, 1, 2, 3, 5, 6]][inds]
        GT[name] = gt

    pre_bboxes = joblib.load('/home/zhai/PycharmProjects/Demo35/SSD/test/voc_MAP/SSD300_2x.pkl')
    AP=mAP(pre_bboxes,GT,20,use_07_metric=False)
    print(AP)
    print(AP.mean())
    pass
