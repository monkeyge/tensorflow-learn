import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #隐藏告警信息
hello = tf.constant('hello tensorflow')
sess = tf.Session()
print(sess.run(hello))
