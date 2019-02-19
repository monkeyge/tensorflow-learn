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

# 丢弃是由Hinton及其在多伦多大学的学生开发的，是最有效且最常用的神经网络正则化技术之一。
# 丢弃（应用于某个层）是指在训练期间随机“丢弃”（即设置为0）该层的多个输出特征。
# 假设某个指定的层通常会在训练期间针对给定的输入样本返回一个向量[0.2,0.5,1.3,0.8,1.1];
# 在应用丢弃后，此向量将随机分布几个0条目，例如[0,0.5,1.3,0,1.1]。“丢弃率”指变为0的特征所占的比例，
# 通常设置在0.2和0.5之间。在测试时，网络不会丢弃任何单元，而是将层的输出值按等同于丢弃率的比例进行缩减，
# 以便平衡以下事实：测试时的活跃单元数大于训练时的活跃单元数。
# 在tf.keras中，您可以通过丢弃层将丢弃引入网络中，以便事先将其应用于层的输出。
# 下面我们在IMDB网络中添加两个丢弃层，看看它们在降低过拟合方面表现如何：
dpt_model = keras.models.Sequential([
    keras.layers.Dense(16, activation=tf.nn.relu, input_shape=(NUM_WORDS,)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(16, activation=tf.nn.relu),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(1, activation=tf.nn.sigmoid)
])

dpt_model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy','binary_crossentropy'])

dpt_model_history = dpt_model.fit(train_data, train_labels,
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
              ('dropout', dpt_model_history)])

# 防止神经网络过拟合常用的方法
# 1.获取更多训练数据。
# 2.降低网络容量。
# 3.添加权重正则化。
# 4.添加丢弃层。
