#-*- encoding=utf8 -*-
from __future__ import absolute_import, division, print_function

import os

import  tensorflow as tf
from tensorflow import keras

print(tf.__version__)

# 加载mnist数据
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()
# 截取前1000个样本
train_labels = train_labels[:1000]
test_labels = test_labels[:1000]
# 二维数组转换成 28 * 28 的一维数组 并且数组值控制在0到1范围内
train_images = train_images[:1000].reshape(-1, 28 * 28) / 255.0
test_images = test_images[:1000].reshape(-1, 28 * 28) / 255.0


# 创建模型
def create_model():
    model = tf.keras.models.Sequential([
        # 隐藏层 512个隐藏单元
        keras.layers.Dense(512, activation=tf.nn.relu, input_shape=(784,)),
        # 丢弃层 随机丢弃20%上层输入样本向量中的数据
        # 即向量中的某些数据设置成0
        keras.layers.Dropout(0.2),
        # 输出层 返回softmax概率分布
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])
    # AdamOptimizer 梯度下降算法优化
    # sparse_categorical_crossentropy 损失函数评估模型准确率
    # accuracy 指标监控
    model.compile(optimizer=tf.keras.optimizers.Adam(),
                  loss=tf.keras.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

    return model


# Create a basic model instance
# model = create_model()
# print(model.summary())

# 检查点回调设置
# 该回调提供了多个选项，用于为生成的检查点提供独一无二的名称，以及调整检查点创建频率
# 训练一个新模型，每隔 5 个周期保存一次检查点并设置唯一名称
proName = "tensorflow-learn-git"
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = curPath[:curPath.find(proName)+len(proName)]  # 获取myProject，也就是项目的根路径
# 检查点保存路径
checkpoint_path = rootPath + "/data/save/cp2/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# 创建检查点
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path, verbose=1, save_weights_only=True,
    # Save weights, every 5-epochs.
    period=5)
# 创建模型
model = create_model()
# 训练模型并且在每个周期结束时更新检查点cp_callback
model.fit(train_images, train_labels,
          epochs = 50, callbacks = [cp_callback],
          validation_data = (test_images,test_labels),
          verbose=0)

print(checkpoint_dir)

# 上述代码将权重存储在检查点格式的文件集合中，
# 这些文件仅包含经过训练的权重（采用二进制格式）。
# 检查点包括： * 包含模型权重的一个或多个分片。
#  * 指示哪些权重存储在哪些分片中的索引文件。
# 如果您仅在一台机器上训练模型，则您将有 1 个后缀为 .data-00000-of-00001 的分片

