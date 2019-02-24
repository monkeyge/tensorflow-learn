import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# 下载IMDB文本影评数据
imdb_data = keras.datasets.imdb
# num_word=10000 保留出现词频在前10000位的词 舍弃罕见次
(train_data, train_labels), (test_data, test_labels) = imdb_data.load_data(num_words=10000)
print("训练数据：{}，标签：{}".format(len(train_data), len(train_labels)))
# 影评文本已经处理成整数
print(train_data[0])
# 影评长度不一样但是神经网络的输入必须长度相同
print(len(train_data[0]))
print(len(train_data[1]))
