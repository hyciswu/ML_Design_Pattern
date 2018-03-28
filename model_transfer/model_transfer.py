from tensorgraph.layers import BaseModel, Conv2D, BatchNormalization, RELU, \
                               Dropout, Flatten, Softmax, AvgPooling, IdentityBlock, \
                               DenseBlock, Linear, DenseNet, MaxPooling, UNet, DenseNet
import tensorgraph as tg
from tensorgraph.utils import same, valid
import tensorflow as tf
import numpy as np


class TemplateModel(BaseModel):

    @BaseModel.init_name_scope
    def __init__(self, nclass, h, w, c):
        layers = []
        # model = DenseNet(input_channels=c, input_shape=(h, w), ndense=1, growth_rate=1, nlayer1blk=1)
        model = DenseNet(input_channels=c, input_shape=(h,w), ndense=3, growth_rate=4, nlayer1blk=4)
        layers.append(model)

        self.output_dim = np.prod(model.output_shape) * model.output_channels
        # import pdb; pdb.set_trace()
        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])



class OldModel(BaseModel):

    def __init__(self, nclass, h, w, c):
        layers = []
        template = TemplateModel(nclass, h, w, c)
        layers.append(template)
        layers.append(Flatten())
        layers.append(Linear(template.output_dim, 100))
        layers.append(RELU())
        layers.append(Linear(100, nclass))
        layers.append(Softmax())

        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])



class NewModel(BaseModel):
    def __init__(self, nclass, h, w, c):
        layers = []
        template = TemplateModel(nclass, h, w, c)
        layers.append(template)
        layers.append(Flatten())
        layers.append(Linear(template.output_dim, 200))
        layers.append(RELU())
        layers.append(Linear(200, nclass))
        layers.append(Softmax())

        self.startnode = tg.StartNode(input_vars=[None])
        model_hn = tg.HiddenNode(prev=[self.startnode], layers=layers)
        self.endnode = tg.EndNode(prev=[model_hn])
