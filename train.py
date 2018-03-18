
import tensorgraph as tg
import numpy as np
import tensorflow as tf
import tensorflow.contrib as tc
import horovod.tensorflow as hvd
# import cifar10_allcnn
from tensorflow.python.framework import ops
from data import cifar10, mnist
from model import *
import logging, os
from config import __MODEL_VARSCOPE__


logging.basicConfig(format='%(module)s.%(funcName)s %(lineno)d:%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
hvd.init()

def train(model, data, epoch_look_back=5, max_epoch=100, percent_decrease=0, batch_size=64,
          learning_rate=0.001, weight_regularize=True, save_dir=None):

    if save_dir and hvd.rank() == 0:
        logdir = '{}/log'.format(save_dir)
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        model_dir = "{}/model".format(save_dir)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    train_tf, n_train, valid_tf, n_valid = data(create_tfrecords=True, batch_size=batch_size)

    y_train_sb = model._train_fprop(train_tf['X'])
    y_valid_sb = model._test_fprop(valid_tf['X'])

    loss_train_sb = tg.cost.mse(y_train_sb, train_tf['y'])

    if weight_regularize:
        loss_reg = tc.layers.apply_regularization(
                        tc.layers.l2_regularizer(2.5e-5),
                        weights_list=[var for var in tf.global_variables() if __MODEL_VARSCOPE__ in var.name]
                        )
        loss_train_sb = loss_train_sb + loss_reg

    accu_train_sb = tg.cost.accuracy(y_train_sb, train_tf['y'])
    # accu_valid_sb = tg.cost.accuracy(y_valid_sb, valid_tf['y'])

    tf.summary.scalar('train', accu_train_sb)

    if save_dir and hvd.rank() == 0:
        saver = tf.train.Saver()

    opt = tf.train.RMSPropOptimizer(learning_rate)
    opt = hvd.DistributedOptimizer(opt)

    # required for BatchNormalization layer
    update_ops = ops.get_collection(ops.GraphKeys.UPDATE_OPS)
    with ops.control_dependencies(update_ops):
        train_op = opt.minimize(loss_train_sb)

    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    bcast = hvd.broadcast_global_variables(0)

    # Pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    with tf.Session(config=config) as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        sess.run(init_op)
        train_writer = tf.summary.FileWriter( '{}/train'.format(logdir), sess.graph)
        bcast.run()
        # merge = tf.summary.merge_all()
        es = tg.EarlyStopper(max_epoch, epoch_look_back, percent_decrease)
        epoch = 0
        best_valid_accu = 0
        while True:
            epoch += 1

            pbar = tg.ProgressBar(n_train)
            ttl_train_loss = 0
            for i in range(0, n_train, batch_size):
                pbar.update(i)
                _, loss_train = sess.run([train_op, loss_train_sb])
                # _, loss_train, merge_v = sess.run([train_op, loss_train_sb, merge])
                ttl_train_loss += loss_train * batch_size
                # train_writer.add_summary(merge_v, i)
            pbar.update(n_train)
            ttl_train_loss /= n_train
            print('')
            logger.info('gpu {}: epoch {}, train loss {}'.format(hvd.rank(), epoch, ttl_train_loss))

            pbar = tg.ProgressBar(n_valid)
            ttl_valid_accu = 0
            for i in range(0, n_valid, batch_size):
                pbar.update(i)
                loss_accu = sess.run(accu_valid_sb)
                ttl_valid_accu += loss_accu * batch_size
            pbar.update(n_valid)
            ttl_valid_accu /= n_valid
            print('')
            logger.info('gpu {}: epoch {}, valid accuracy {}'.format(hvd.rank(), epoch, ttl_valid_accu))
            if es.continue_learning(-ttl_valid_accu, epoch=epoch):
                logger.info('gpu {}: best epoch last update: {}'.format(hvd.rank(), es.best_epoch_last_update))
                logger.info('gpu {}: best valid last update: {}'.format(hvd.rank(), es.best_valid_last_update))

                if ttl_valid_accu > best_valid_accu:
                    best_valid_accu = ttl_valid_accu
                    if save_dir and hvd.rank() == 0:
                        save_path = saver.save(sess, model_dir + '/model.tf')
                        print("Best model saved in file: %s" % save_path)

            else:
                logger.info('gpu {}: training done!'.format(hvd.rank()))
                break

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    with tf.Graph().as_default():
        with tf.variable_scope(__MODEL_VARSCOPE__):
            # model = AllCNNPlus(nclass=10, h=28, w=28, c=1)
            # model = ResDense(nclass=10, h=28, w=28, c=1)
            # model = DenseNetModel(nclass=10, h=28, w=28, c=1)
            model = UNetModel(nclass=10, h=32, w=32, c=3)

        # timestamp
        ts = tg.utils.ts()
        save_dir='./save/{}'.format(ts)
        train(model, cifar10, epoch_look_back=5, max_epoch=100, percent_decrease=0,
              batch_size=64, learning_rate=0.001, weight_regularize=True,
              save_dir=save_dir)
