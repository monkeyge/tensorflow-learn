import tensorflow as tf
from tensorflow import keras

import matplotlib.pyplot as plt
import numpy as np

print(tf.__version__)

# 下载IMDB文本影评数据
imdb_data = keras.datasets.imdb
NUM_WORDS = 10000
# num_word=10000 保留出现词频在前10000位的词 舍弃罕见次
(train_data, train_labels), (test_data, test_labels) = imdb_data.load_data(num_words=NUM_WORDS)


def multi_hot_sequences(sequences, dimension):
    # Create an all-zero matrix of shape (len(sequences), dimension)
    results = np.zeros((len(sequences), dimension))
    for i, word_indices in enumerate(sequences):
        results[i, word_indices] = 1.0  # set specific indices of results[i] to 1s
    return results


train_data = multi_hot_sequences(train_data, dimension=NUM_WORDS)
print(train_data[0])
test_data = multi_hot_sequences(test_data, dimension=NUM_WORDS)

plt.plot(train_data[0])
plt.show()
