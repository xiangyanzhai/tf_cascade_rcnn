# !/usr/bin/python
# -*- coding:utf-8 -*-

class Config():
    def __init__(self, is_train, Mean, files, lr=0.00125, weight_decay=0.0001,
                 num_cls=80, img_max=1333,
                 img_min=800, anchor_scales=[[32], [64], [128], [256], [512]],
                 anchor_ratios=[[0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2], [0.5, 1, 2]],
                 batch_size_per_GPU=1, gpus=1,
                 rpn_n_sample=256,
                 rpn_pos_iou_thresh=0.7, rpn_neg_iou_thresh=0.3,
                 rpn_pos_ratio=0.5,
                 roi_nms_thresh=0.7,
                 roi_train_pre_nms=12000,
                 roi_train_post_nms=2000,
                 roi_test_pre_nms=6000,
                 roi_test_post_nms=1000,
                 roi_min_size=[4, 8, 16, 32, 64],
                 fast_n_sample=512,
                 fast_pos_ratio=0.25, fast_pos_iou_thresh=0.5,
                 fast_neg_iou_thresh_hi=0.5, fast_neg_iou_thresh_lo=0.0,

                 ):
        self.is_train = is_train
        self.Mean = Mean
        self.files = files
        self.lr = lr
        self.weight_decay = weight_decay
        self.num_cls = num_cls
        self.img_max = img_max
        self.img_min = img_min
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.batch_size_per_GPU = batch_size_per_GPU
        self.gpus = gpus
        self.rpn_n_sample = rpn_n_sample
        self.rpn_pos_iou_thresh = rpn_pos_iou_thresh
        self.rpn_neg_iou_thresh = rpn_neg_iou_thresh
        self.rpn_pos_ratio = rpn_pos_ratio
        self.roi_nms_thresh = roi_nms_thresh
        self.roi_train_pre_nms = roi_train_pre_nms
        self.roi_train_post_nms = roi_train_post_nms
        self.roi_test_pre_nms = roi_test_pre_nms
        self.roi_test_post_nms = roi_test_post_nms
        self.roi_min_size = roi_min_size
        self.fast_n_sample = fast_n_sample
        self.fast_pos_ratio = fast_pos_ratio
        self.fast_pos_iou_thresh = fast_pos_iou_thresh
        self.fast_neg_iou_thresh_hi = fast_neg_iou_thresh_hi
        self.fast_neg_iou_thresh_lo = fast_neg_iou_thresh_lo

        # print('Mask_Rcnn')
        print('==============================================================')
        print('Mean:\t', self.Mean)
        print('files:\t', self.files)
        print('lr:\t', self.lr)
        print('weight_decay:\t', self.weight_decay)
        print('num_cls:\t', self.num_cls)
        print('img_max:\t', self.img_max)
        print('img_min:\t', self.img_min)
        print('anchor_scales:\t', self.anchor_scales)
        print('anchor_ratios:\t', self.anchor_ratios)
        print('batch_size_per_GPU:\t', self.batch_size_per_GPU)
        print('gpus:\t', self.gpus)
        print('lr*batch_size_per_GPU*gpus:\t', self.lr * self.batch_size_per_GPU * self.gpus)
        print('==============================================================')
        print('rpn_n_sample:\t', self.rpn_n_sample)
        print('rpn_pos_iou_thresh:\t', self.rpn_pos_iou_thresh)
        print('rpn_neg_iou_thresh:\t', self.rpn_neg_iou_thresh)
        print('rpn_pos_ratio:\t', self.rpn_pos_ratio)
        print('==============================================================')
        print('roi_nms_thresh:\t', self.roi_nms_thresh)
        print('roi_train_pre_nms:\t', self.roi_train_pre_nms)
        print('roi_train_post_nms:\t', self.roi_train_post_nms)
        print('roi_test_pre_nms:\t', self.roi_test_pre_nms)
        print('roi_test_post_nms:\t', self.roi_test_post_nms)
        print('roi_min_size :\t', self.roi_min_size)
        print('==============================================================')
        print('fast_n_sample :\t', self.fast_n_sample)
        print('fast_pos_ratio :\t', self.fast_pos_ratio)
        print('fast_pos_iou_thresh :\t', self.fast_pos_iou_thresh)
        print('fast_neg_iou_thresh_hi :\t', self.fast_neg_iou_thresh_hi)
        print('fast_neg_iou_thresh_lo  :\t', self.fast_neg_iou_thresh_lo)
        print('==============================================================')

        pass
