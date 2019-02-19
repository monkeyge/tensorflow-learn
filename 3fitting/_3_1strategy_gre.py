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
# 创建基准模型
baseline_model = keras.Sequential([
    # `input_shape` is only required here so that `.summary` works.
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])
baseline_model.compile(optimizer='adam',
                       loss='binary_crossentropy',
                       metrics=['accuracy', 'binary_crossentropy'])
print(baseline_model.summary())
# 获得训练历史
baseline_history = baseline_model.fit(train_data,
                                      train_labels,
                                      epochs=20,
                                      batch_size=512,
                                      validation_data=(test_data, test_labels),
                                      verbose=2)

# 创建L2正则化模型
l2_model = keras.models.Sequential([
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dense(16, kernel_regularizer=keras.regularizers.l2(0.001),
                       activation=tf.nn.relu),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

l2_model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy', 'binary_crossentropy'])

l2_model_history = l2_model.fit(train_data, train_labels,
                                epochs=20,
                                batch_size=512,
                                validation_data=(test_data, test_labels),
                                verbose=2)


# 绘制训练损失和验证损失图表
# 实线表示训练损失，虚线表示验证损失（请谨记：验证损失越低，表示模型越好）。
# 在此示例中，较小的网络开始过拟合的时间比基准模型晚（前者在6个周期之后，后者在4个周期之后），
# 并且开始过拟合后，它的效果下降速度也慢得多
# 较大的网络几乎仅仅1个周期之后便立即开始过拟合，并且之后严重得多。
# 网络容量越大，便能够越快对训练数据进行建模（产生较低的训练损失），
# 但越容易过拟合（导致训练损失与验证损失之间的差异很大）
def plot_history(histories, key='binary_crossentropy'):
    plt.figure(figsize=(16, 10))

    for name, history in histories:
        val = plt.plot(history.epoch, history.history['val_'+key],
                       '--', label=name.title()+' Val')
        plt.plot(history.epoch, history.history[key], color=val[0].get_color(),
                 label=name.title()+' Train')

    plt.xlabel('Epochs')
    plt.ylabel(key.replace('_', ' ').title())
    plt.legend()

    plt.xlim([0, max(history.epoch)])
    plt.show()


plot_history([('baseline', baseline_history),
              ('L2', l2_model_history)])
