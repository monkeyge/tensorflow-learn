# -*- coding: utf-8 -*-

from __future__ import absolute_import, division, print_function
# 面向对象的文件系统路径
import pathlib
# 引入数据分析包
import pandas as pd
# Seaborn其实是在matplotlib的基础上进行了更高级的API封装，
# 从而使得作图更加容易，在大多数情况下使用seaborn就能做出很具有吸引力的图，
# 而使用matplotlib就能制作具有更多特色的图。
# 应该把Seaborn视为matplotlib的补充，而不是替代物。
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from pylab import *

# 处理话图表的中文
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
# 预测燃油效率
print(tf.__version__)

dataset_path = keras.utils.get_file("auto-mpg.data", "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
print("数据路径："+dataset_path)

# 定义列名称
column_names = ['MPG', '气缸', '排量', '马力', '重量',
                '加速', '年份', 'Origin']
# column_names = ['MPG','Cylinders','Displacement','Horsepower','Weight',
#                 'Acceleration', 'Model Year', 'Origin']
# skipinitialspace 忽略分隔符后的空白（默认为False，即不忽略）
raw_dataset = pd.read_csv(dataset_path, names=column_names,
                          na_values="?", comment='\t',
                          sep=" ", skipinitialspace=True)
dataset = raw_dataset.copy()
print(dataset.tail())

# 查看数据里面是否包含未知值
# print(dataset.isna().sum())

# 清理未知值
dataset = dataset.dropna()
# print(dataset.isna().sum())

# 将origin列的数据转换
origin = dataset.pop('Origin')
dataset['美国'] = (origin == 1)*1.0
dataset['欧洲'] = (origin == 2)*1.0
dataset['日本'] = (origin == 3)*1.0
print(dataset.tail())

# 将数据分为训练集合测试集
train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_stats = train_dataset.describe()
train_stats.pop("MPG")
# 行转列统计查看
train_stats = train_stats.transpose()

# 分离出标签
train_labels = train_dataset.pop('MPG')
test_labels = test_dataset.pop('MPG')
print(test_labels)


# 规范数据
def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


normed_train_data = norm(train_dataset)
normed_test_data = norm(test_dataset)


# 创建模型
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)
    # MSE 均方误差损失函数（回归、分类问题）
    # MAE 平均绝对误差（度量回归）
    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


model = build_model()
print(model.summary())

example_batch = normed_train_data[:10]
example_result = model.predict(example_batch)
print(example_result)


# 训练模型
# Display training progress by printing a single dot for each completed epoch
class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0:print('')
        print('.', end='')


EPOCHS = 1000
# 训练模型1000个时期，并记录对象的训练和验证准确性history
# model.fit当验证分数没有改善时，让我们更新呼叫以自动停止训练。
# 我们将使用EarlyStopping回调来测试每个时期的训练条件。
# 如果经过一定数量的时期而没有显示出改进，则自动停止训练。
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
history = model.fit(normed_train_data, train_labels, epochs=EPOCHS,
                    validation_split=0.2, verbose=0, callbacks=[early_stop, PrintDot()])
hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.tail())


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 5])

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],
             label='Val Error')
    plt.legend()
    plt.ylim([0, 20])
    plt.show()


plot_history(history)
loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=0)
print("误差: {:5.2f} MPG".format(mae))

# 测试数据预测MPG值
test_predictions = model.predict(normed_test_data).flatten()

plt.scatter(test_labels, test_predictions)
plt.xlabel('真实值 [MPG]')
plt.ylabel('预测值 [MPG]')
plt.axis('equal')
plt.axis('square')
plt.xlim([0, plt.xlim()[1]])
plt.ylim([0, plt.ylim()[1]])
_ = plt.plot([-100, 100], [-100, 100])
plt.show()

# 查看错误分布
error = test_predictions - test_labels
plt.hist(error, bins = 25)
plt.xlabel("Prediction Error [MPG]")
_ = plt.ylabel("Count")
plt.show()

