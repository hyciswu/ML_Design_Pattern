
from tensorgraph.layers import BaseModel, Conv2D, BatchNormalization, RELU, \
                               Dropout, Flatten, Softmax, AvgPooling, IdentityBlock, \
                               DenseBlock, Linear, DenseNet, MaxPooling, UNet
import tensorgraph as tg
from tensorgraph.utils import same, valid
import tensorflow as tf


class ResDense(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nclass, h, w, c):
        layers = []
        with tf.name_scope('blk1'):
            identityblk = IdentityBlock(input_channels=c, input_shape=[h,w], nlayers=10)
            layers.append(identityblk)

            layers.append(Conv2D(input_channels=c, num_filters=16, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
            layers.append(RELU())
            h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
            layers.append(BatchNormalization(input_shape=[h,w,16]))

        with tf.name_scope('blk2'):
            denseblk = DenseBlock(input_channels=16, input_shape=[h,w], growth_rate=4, nlayers=4)
            layers.append(denseblk)

            layers.append(Conv2D(input_channels=denseblk.output_channels, num_filters=32, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
            layers.append(RELU())
            h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
            layers.append(Dropout(0.5))

        with tf.name_scope('blk3'):
            layers.append(Conv2D(input_channels=32, num_filters=nclass, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
            layers.append(RELU())
            h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
            layers.append(BatchNormalization(input_shape=[h,w,nclass]))

        layers.append(AvgPooling(poolsize=(h, w), stride=(1,1), padding='VALID'))
        layers.append(Flatten())
        layers.append(Softmax())

        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])


class AllCNNPlus(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nclass, h, w, c):
        layers = []
        layers.append(AllCNN(nclass, h, w, c))
        layers.append(Linear(nclass, nclass))
        layers.append(Softmax())
        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])


class AllCNN(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nclass, h, w, c):
        layers = []
        layers.append(Conv2D(input_channels=c, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        layers.append(BatchNormalization(input_shape=[h,w,96]))

        layers.append(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        layers.append(Dropout(0.5))

        layers.append(Conv2D(input_channels=96, num_filters=96, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
        layers.append(BatchNormalization(input_shape=[h,w,96]))

        layers.append(Conv2D(input_channels=96, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        layers.append(Dropout(0.5))

        layers.append(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        layers.append(BatchNormalization(input_shape=[h,w,192]))

        layers.append(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(2, 2), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(2,2), kernel_size=(3,3))
        layers.append(Dropout(0.5))

        layers.append(Conv2D(input_channels=192, num_filters=192, kernel_size=(3, 3), stride=(1, 1), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(3,3))
        layers.append(BatchNormalization(input_shape=[h,w,192]))

        layers.append(Conv2D(input_channels=192, num_filters=192, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
        layers.append(Dropout(0.5))

        layers.append(Conv2D(input_channels=192, num_filters=nclass, kernel_size=(1, 1), stride=(1, 1), padding='SAME'))
        layers.append(RELU())
        h, w = same(in_height=h, in_width=w, stride=(1,1), kernel_size=(1,1))
        layers.append(BatchNormalization(input_shape=[h,w,nclass]))

        layers.append(AvgPooling(poolsize=(h, w), stride=(1,1), padding='VALID'))
        layers.append(Flatten())
        layers.append(Softmax())
        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])


class DenseNetModel(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nclass, h, w, c):
        layers = []
        model = DenseNet(input_channels=c, input_shape=(h, w), ndense=1, growth_rate=1, nlayer1blk=1)
        layers.append(model)
        layers.append(MaxPooling(poolsize=tuple(model.output_shape), stride=(1,1), padding='VALID'))
        layers.append(Flatten())
        layers.append(Linear(model.output_channels, nclass))
        layers.append(Softmax())
        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])


class UNetModel(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nclass, h, w, c):
        layers = []
        model = UNet(input_channels=c, input_shape=(h, w))
        layers.append(model)
        layers.append(MaxPooling(poolsize=tuple(model.output_shape), stride=(1,1), padding='VALID'))
        layers.append(Flatten())
        layers.append(Linear(model.output_channels, nclass))
        layers.append(Softmax())
        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])
