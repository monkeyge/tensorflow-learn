#-*- encoding=utf8 -*-
from __future__ import absolute_import, division, print_function

import os

import  tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# 加载mnist数据
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# 截取前1000个样本
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
# 二维数组转换成 28 * 28 的一维数组 并且数组值控制在0到1范围内
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0

