import tensorflow as tf
import numpy as np
import scipy.misc
from canny import *

class canny_or(object):
    def __init__(self):
        pass
    def model(self):
        print("model")
        self.image = tf.placeholder(tf.float32, [None,32,32,3],name="image")
        self.canny = canny(self.image, low=0.10 * 255., high=0.20 * 255.)

    def train(self):
        print("train")
        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)
        Lena = scipy.misc.imread("/home/guangyuan/conclusion/canny-tensorflow/07.jpg")\
            .reshape(1, 32, 32, 3).astype(np.float32)
        canny = sess.run(self.canny, feed_dict={self.image: Lena}).reshape(32, 32)
        scipy.misc.imsave("/home/guangyuan/conclusion/canny-tensorflow/c07.jpg", canny)

canny_=canny_or()
canny_.model()
canny_.train()

#CUDA_VISIBLE_DEVICES=0 python test.py