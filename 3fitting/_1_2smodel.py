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

# plt.plot(train_data[0])
# plt.show()

# 深度学习模型往往善于与训练数据拟合，但真正的挑战是泛化，而非拟合。
# 创建更小的模型
smaller_model = keras.Sequential([
    keras.layers.Dense(4, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(4, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

smaller_model.compile(optimizer='adam',
                      loss='binary_crossentropy',
                      metrics=['accuracy', 'binary_crossentropy'])

smaller_model.summary()
# 训练模型
smaller_history = smaller_model.fit(train_data,
                                    train_labels,
                                    epochs=20,
                                    batch_size=512,
                                    validation_data=(test_data, test_labels),
                                    verbose=2)
