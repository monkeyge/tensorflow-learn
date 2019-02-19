import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

# 下载IMDB文本影评数据
imdb_data = keras.datasets.imdb
# num_word=10000 保留出现词频在前10000位的词 舍弃罕见次
(train_data, train_labels), (test_data, test_labels) = imdb_data.load_data(num_words=10000)
# 获得字典索引
word_index = imdb_data.get_word_index()
print("key:{},value:{}".format("good", word_index.get("good")))
word_index = {k: (v+3) for k, v in word_index.items()}
print("key:{},value:{}".format("good", word_index.get("good")))
# 占位符处理
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3
# 字典key value 互换
receive_word_index = dict([(value, key) for (key, value) in word_index.items()])
# print(receive_word_index)


def decode_review(data):
    return ' '.join([receive_word_index.get(i, '?') for i in data])


text = decode_review(train_data[0])
print(text)

