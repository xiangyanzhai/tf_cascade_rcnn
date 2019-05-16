# !/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
def generate_anchor_base(anchor_scales, anchors_ratios, base_size=16):
    py = base_size / 2.
    px = base_size / 2.
    anchor_base = np.zeros((len(anchors_ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in range(len(anchors_ratios)):
        for j in range(len(anchor_scales)):
            h = anchor_scales[j] * np.sqrt(anchors_ratios[i])
            w = anchor_scales[j] * np.sqrt(1. / anchors_ratios[i])
            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
def get_coord(N):
    t = np.arange(N)
    x, y = np.meshgrid(t, t)
    x = x[..., None]
    y = y[..., None]
    coord = np.concatenate((y, x, y, x), axis=-1)
    coord = coord[:, :, None, :]
    return coord


def get_anchors(N, anchor_scales, anchors_ratios,stride=16 ):
    coord = get_coord(N)
    anchor_base = generate_anchor_base(anchor_scales, anchors_ratios,base_size=stride)
    anchors = coord * stride + anchor_base
    print(anchor_base)
    return tf.constant(anchors,dtype=tf.float32)

def py_get_anchors(N, anchor_scales, anchors_ratios,stride=16 ):
    coord = get_coord(N)
    anchor_base = generate_anchor_base(anchor_scales, anchors_ratios,base_size=stride)
    anchors = coord * stride + anchor_base
    print(anchor_base)
    return anchors


if __name__ == "__main__":
    pass