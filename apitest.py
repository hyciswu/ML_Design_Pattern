
from tensorgraph.dataset import Mnist
from api import AllCNNPlusPredict
import tensorgraph as tg
import numpy as np
from model import AllCNNPlus
from train import train
from data import mnist
from config import __MODEL_VARSCOPE__
import tensorflow as tf

def train_test():
    '''test runtime'''
    with tf.Graph().as_default():
        with tf.variable_scope(__MODEL_VARSCOPE__):
            model = AllCNNPlus(nclass=10, h=28, w=28, c=1)

        train(model, mnist, epoch_look_back=5, max_epoch=1, percent_decrease=0,
              batch_size=64, learning_rate=0.001, weight_regularize=True,
              save_dir=None)
    print('runtime test passed!')


def predict_test():
    X_train, y_train, X_test, y_test = tg.dataset.Mnist()
    model_path = './save/20180222_1519_54083986/model/model.tf'
    model_trained = AllCNNPlusPredict(model_path=model_path, nclass=10, h=28, w=28, c=1)
    n_exp = 1000
    y_pred = model_trained.predict(X_test[:n_exp])
    y_pred = np.argmax(y_pred, 1)
    y_test = np.argmax(y_test[:n_exp], 1)
    accu = np.mean(y_pred == y_test)
    print('test accuracy:', accu)
    assert accu > 0.95
    print('predict test passed!')

if __name__ == '__main__':
    train_test()
    predict_test()
