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


def plot_image(i, predictions_array, true_label, img):
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

# 绘制其分类和比例以及对应的图片
# i = 12
# plt.figure(figsize=(6, 3))
# plt.subplot(1, 2, 1)
# plot_image(i, predictions, test_labels, test_images)
# plt.subplot(1, 2, 2)
# plot_value_array(i, predictions,  test_labels)
# plt.show()

# 绘制多张图片
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_image(i, predictions, test_labels, test_images)
    plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_value_array(i, predictions, test_labels)
plt.show()
