# -*- coding: utf-8 -*-

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras

import _0fileload

print(tf.__version__)

(train_images, train_labels), (test_images, test_labels) = _0fileload.load_file()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(len(train_images))
print(test_images.shape)
print(len(test_labels))

# plt.figure(figsize=(5, 5))
# plt.imshow(train_images[3])
# plt.colorbar()
# plt.grid(False)
# plt.show()
train_images = train_images / 255

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
