# -*- coding: utf-8 -*-

import os
import numpy as np
import struct
import matplotlib.pyplot as plt


def read_image(file_name):
    '''
    :param file_name: 文件路径
    :return:  训练或者测试数据
    如下是训练的图片的二进制格式
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000803(2051) magic number
    0004     32 bit integer  60000            number of images
    0008     32 bit integer  28               number of rows
    0012     32 bit integer  28               number of columns
    0016     unsigned byte   ??               pixel
    0017     unsigned byte   ??               pixel
    ........
    xxxx     unsigned byte   ??               pixel
    '''
    file_handle = open(file_name , "rb")  #以二进制打开文档
    file_content = file_handle.read()   #读取到缓冲区中
    head = struct.unpack_from('>IIII', file_content, 0)  # 取前4个整数，返回一个元组
    offset = struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    width = head[2]   #宽度
    height = head[3]  #高度
    bits = imgNum * width * height  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
    imgs = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    imgs_array=np.array(imgs).reshape((imgNum,width*height))     #最后将读取的数据reshape成 【图片数，图片像素】二维数组
    return imgs_array


def out_image(img):
    '''
    :param img: 图片像素组
    :return:
    '''
    plt.figure()
    plt.imshow(img)
    plt.show()


def out_image_lable(img,lable_name,class_names):
    '''
    :param img: 图片像素组
    :return:
    '''
    img = img / 255.0

    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(img, cmap=plt.cm.binary)
    plt.xlabel(class_names[lable_name])


def read_label(file_name):
    '''
    :param file_name:
    :return:
    标签的格式如下：
    [offset] [type]          [value]          [description]
    0000     32 bit integer  0x00000801(2049) magic number (MSB first)
    0004     32 bit integer  60000            number of items
    0008     unsigned byte   ??               label
    0009     unsigned byte   ??               label
    ........
    xxxx     unsigned byte   ??               label
    The labels values are 0 to 9.
    '''
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中
    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')
    labelNum = head[1]  # label数
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)


def get_data():
    # 文件获取
    proName = "tensorflow-learn"
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find(proName)+len(proName)]  # 获取myProject，也就是项目的根路径
    base_path = rootPath + "/data/fashion/"
    # fashion-mnist数据
    train_image = base_path + "train-images-idx3-ubyte"
    test_image = base_path + "t10k-images-idx3-ubyte"
    train_label = base_path + "train-labels-idx1-ubyte"
    test_label = base_path + "t10k-labels-idx1-ubyte"

    # MNIST数据
    # base_path = rootPath + "/data/mnist/"
    # train_image = base_path + "train-images.idx3-ubyte"
    # test_image = base_path + "t10k-images.idx3-ubyte"
    # train_label = base_path + "train-labels.idx1-ubyte"
    # test_label = base_path + "t10k-labels.idx1-ubyte"
    # 读取数据
    train_x = read_image(train_image)
    test_x = read_image(test_image)
    train_y = read_label(train_label)
    test_y = read_label(test_label)
    print(train_y[0:10])
    print(test_y[0:10])
    #显示一张图片
    # out_image(np.array(train_x[0]).reshape(28, 28))
    #显示多张图片
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        out_image_lable(np.array(train_x[i]).reshape(28, 28), train_y[i],class_names)
    plt.show()
    return train_x, train_y, test_x, test_y


if __name__ == "__main__":
    get_data()
