

import tensorflow as tf
import model
from config import __MODEL_VARSCOPE__

class AllCNNPlusPredict(object):


    def __init__(self, model_path, nclass, h, w, c):
        '''
        shape (list of 3 ints): [height, width, channel]
        '''

        self.graph = tf.Graph()
        # self.preprocessor = preprocessor
        # self.shape = list(shape)

        with self.graph.as_default():

            with tf.variable_scope(__MODEL_VARSCOPE__):
                self.model = model.AllCNNPlus(nclass=10, h=28, w=28, c=1)

            self.sess = tf.Session()
            init_op = tf.group(tf.global_variables_initializer(),
                               tf.local_variables_initializer())

            self.sess.run(init_op)
            saver = tf.train.Saver()
            saver.restore(self.sess, model_path)

            self.X_ph = tf.placeholder('float32', [None] + [h, w, c])
            self.y_test_sb = self.model._test_fprop(self.X_ph)


    def predict(self, X):
        '''
        X (4 dim npy array): [batch, height, width, channel]
        '''
        with self.graph.as_default():
            y = self.sess.run(self.y_test_sb, feed_dict={self.X_ph:X})
        return y
