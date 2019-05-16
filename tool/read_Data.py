# !/usr/bin/python
# -*- coding:utf-8 -*-
import tensorflow as  tf
import numpy as np
import cv2
tf_True=tf.constant(True)
tf_False=tf.constant(False)
print(tf_True,tf_False)
def bbox_flip_left_right(bboxes, w):
    w = tf.to_float(w)
    y1, x1, y2, x2, cls = tf.split(bboxes, 5, axis=1)
    x1, x2 = w - 1. - x2, w - 1. - x1
    return tf.concat([y1, x1, y2, x2, cls], axis=1)

def parse(se):
    features = tf.parse_single_example(
        se, features={
            'im': tf.FixedLenFeature([], tf.string),
            'bboxes': tf.FixedLenFeature([], tf.string),
            'Id': tf.FixedLenFeature([], tf.string),
        }
    )
    img=features['im']
    bboxes = features['bboxes']
    Id = features['Id']
    img = tf.image.decode_jpeg(img, channels=3)
    bboxes = tf.decode_raw(bboxes, tf.float32)
    bboxes = tf.reshape(bboxes, (-1, 5))

    img, bboxes = tf.cond(tf.random_uniform(shape=()) > 0.5, lambda: (
        tf.image.flip_left_right(img), bbox_flip_left_right(bboxes, tf.shape(img)[1])), lambda: (img, bboxes))
    return img, bboxes, tf.shape(bboxes)[0]




def readData(files, batch_size=1, num_epochs=2000, num_threads=16, shuffle_buffer=1024,num_shards=1,shard_index=0):
    dataset = tf.data.TFRecordDataset(files)
    dataset = dataset.shard(num_shards, shard_index)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.map(parse, num_parallel_calls=num_threads)

    if shuffle_buffer is not None:
        dataset = dataset.shuffle(shuffle_buffer)
    dataset = dataset.padded_batch(batch_size, padded_shapes=([None, None, 3], [None, 5], []),
                                   padding_values=(np.array(127, dtype=np.uint8), np.array(0.0, dtype=np.float32),
                                                   np.array(0.0, dtype=np.int32)))
    # dataset = dataset.batch(batch_size)

    iter = dataset.make_initializable_iterator()
    return iter




def draw_gt(im, gt):
    im = im.astype(np.uint8)
    boxes = gt.astype(np.int32)
    for box in boxes:
        # print(box)
        y1, x1, y2, x2 = box[:4]
        print(box)
        im = cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 255))
        print(box[-1])
    im = im.astype(np.uint8)
    cv2.imshow('a', im)
    cv2.waitKey(2000)
    return im


if __name__ == "__main__":
    pass