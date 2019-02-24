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
# from tensorflow.keras import layers
import matplotlib.pyplot as plt

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


