# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np

import _0fileload

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = _0fileload.load_file()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_images))
print(test_images.shape)
print(len(test_labels))

# 将二维数组转换成 28 * 28 的一位数组
# 定义第一层128个节点（神经元）
# 定义softmax 概率分布为10种 相加等于1 的概率分布
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

# AdamOptimizer 梯度下降算法优化
# sparse_categorical_crossentropy 损失函数评估模型zhunql
# accuracy 指标监控
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练
train_images = train_images / 255
model.fit(train_images, train_labels, epochs=5)

#使用模型在测试数据上 评估准确率
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('Test accuracy:', test_loss,test_acc)
# 如果机器学习模型在新数据上的表现不如在训练数据上的表现，就表示出现过拟合

#预测
predictions = model.predict(test_images)
print(predictions[0])
# 获得最大概率的index
class_index = np.argmax(predictions[0])
print(class_index)
print(test_labels[0])
print(class_names[class_index])