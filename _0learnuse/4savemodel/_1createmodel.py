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


# 创建模型
def create_model():
    model = tf.keras.models.Sequential([
        # 隐藏层 512个隐藏单元
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        # 丢弃层 随机丢弃20%上层输入样本向量中的数据
        # 即向量中的某些数据设置成0
        keras.layers.Dropout(0.2),
        # 输出层 返回softmax概率分布
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # AdamOptimizer 梯度下降算法优化
    # sparse_categorical_crossentropy 损失函数评估模型准确率
    # accuracy 指标监控
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


# Create a basic model instance
model = create_model()
print(model.summary())

