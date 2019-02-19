
import os
from tensorflow.python.keras.utils import get_file
import gzip
import numpy as np

def load_file():
    # base = "file:///D:/enviroment/TensorFlow/mnist/"
    # files = [
    #     'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
    #     't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    # ]
    proName = "tensorflow-learn"
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = curPath[:curPath.find(proName)+len(proName)]  # 获取myProject，也就是项目的根路径

    base_path = "file:///" + rootPath + "/data/fashion/"
    files = [
        'train-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz',
        't10k-labels-idx1-ubyte.gz', 't10k-images-idx3-ubyte.gz'
    ]

    paths = []
    for fname in files:
        paths.append(get_file(fname, origin=base_path+fname))

    print(paths)

    with gzip.open(paths[0], 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(),np.uint8, offset=8)
    with gzip.open(paths[1], 'rb') as imgpath:
        x_train = np.frombuffer(imgpath.read(),np.uint8, offset=16).reshape(len(y_train), 28, 28)

    with gzip.open(paths[2], 'rb') as lbpath:
        y_test = np.frombuffer(lbpath.read(),np.uint8, offset=8)
    with gzip.open(paths[3], 'rb') as imgpath:
        x_test = np.frombuffer(imgpath.read(),np.uint8, offset=16).reshape(len(y_test), 28, 28)

    return (x_train, y_train), (x_test, y_test)
