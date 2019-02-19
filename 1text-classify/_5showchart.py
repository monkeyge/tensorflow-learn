# -*- coding: utf-8 -*-

# 参考地址
# https://tensorflow.google.cn/tutorials/keras/basic_text_classification

import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
from pylab import *
import numpy as np

# 处理话图表的中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
print(tf.__version__)

# 下载IMDB文本影评数据
imdb_data = keras.datasets.imdb
# num_word=10000 保留出现词频在前10000位的词 舍弃罕见次
(train_data, train_labels), (test_data, test_labels) = imdb_data.load_data(num_words=10000)
print("训练数据：{}，标签：{}".format(len(train_data), len(train_labels)))
# 影评文本已经处理成整数
print(train_data[0])
# 影评长度不一样但是神经网络的输入必须长度相同
print("长度不同：{}，{}".format(len(train_data[0]), len(train_data[1])))

# 获得字典索引
word_index = imdb_data.get_word_index()
# 占位符处理
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
# 影评数组转换成张量（多维数组）
# 长度标准化后馈送到神经网络
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index["<PAD>"],
                                                        padding='post',
                                                        maxlen= 256)
test_data = keras.preprocessing.sequence.pad_sequences(test_data,
                                                       value=word_index["<PAD>"],
                                                       padding='post',
                                                       maxlen=256)

print("长度相同同：{}，{}".format(len(train_data[0]), len(train_data[1])))
print(train_data[0])

# 构建模型
vocab_size = 10000

model = keras.Sequential()
# 词向量
model.add(keras.layers.Embedding(vocab_size, 16))
model.add(keras.layers.GlobalAveragePooling1D())
# 全连接层dense relu激活函数去线性化 快速找到最优参数
model.add(keras.layers.Dense(16, activation=tf.nn.relu))
# 输出层dense sigmoid激活函数 引入概率做分类用
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid))
# 统计每层的隐藏节点（隐藏单元）和参数
model.summary()

# 使用损失函数和优化器
# AdamOptimizer 梯度下降算法优化
# binary_crossentropy损失函数 测量实际分布和预测之间的差距
model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])
# 创建训练集
# 在训练时，我们需要检查模型处理从未见过的数据的准确率。
# 我们从原始训练数据中分离出 10000 个样本，创建一个验证集。
# （为什么现在不使用测试集？
# 我们的目标是仅使用训练数据开发和调整模型，
# 然后仅使用一次测试数据评估准确率。）
x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

# 训练模型
# 用有 512 个样本的小批次训练模型 40 个周期。
# 这将对 x_train 和 y_train 张量中的所有样本进行 40 次迭代。
# 在训练期间，监控模型在验证集的 10000 个样本上的损失和准确率：
history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)
# 评估模型
results = model.evaluate(test_data, test_labels)
print(results)

# 显示准确率和损失随时间变化的图
# history对象包含一个字典，其中包括训练期间发生的所有情况：
history_dict = history.history
print(history_dict.keys())
# dict_keys(['loss', 'val_loss', 'val_acc', 'acc'])

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

# 创建 1 到 len(acc)+1 的整数列表
epochs = range(1, len(acc)+1)
# -----------------------------画损失随时间的变化
# 蓝色圆点
plt.plot(epochs, loss, 'bo', label='训练损失')
# 蓝色线
plt.plot(epochs, val_loss, 'b', label='校验损失')
plt.title('训练和校验损失')
plt.xlabel('时间')
plt.ylabel('损失')
plt.legend()
plt.show()

# ----------------------------画准确率随时间的变化
plt.clf()   # clear figure
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs, acc, 'bo', label='训练准确率')
plt.plot(epochs, val_acc, 'b', label='校验准确率')
plt.title('训练和校验的准确率')
plt.xlabel('时间')
plt.ylabel('准确率')
plt.legend()

plt.show()
