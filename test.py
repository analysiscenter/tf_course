from dataset import DatasetIndex, Dataset, Pipeline
import numpy as np
import os
import pickle
from cifar_batch import Cifar10Batch
import sys
import tensorflow as tf

from dataset.dataset.models.tf.layers import *
from dataset.dataset.models.tf import *

index = DatasetIndex(np.arange(100))

# aux func for reading all arrays in dict
def read_in_dict(src, components):
    res = dict()
    for comp in components:
        with open(os.path.join(src, comp + '.pkl'), 'rb') as file:
            data = pickle.load(file)

        res.update({comp:data})

    return res

datadict = read_in_dict(src='D:/Work/OpenData/', components=['images', 'labels', 'picnames'])



class SmallResNet(TFModel):
    """ Model-class implementing small resnet.
    """

    @staticmethod
    def identity_block(inputs, kernel_size=(3, 3)):
        """ Build identity block with two convs and skip.
        """
        filters = self.get_shape(inputs)[-1]
        output = conv2d_block(inputs, filters, kernel_size, 'cna', is_training=self.is_training)
        output = conv2d_block(output, filters, kernel_size, 'ca', is_training=self.is_training)
        return output + inputs

    @staticmethod
    def halfing_block(inputs, kernel_size=(3, 3)):
        """ Build downsampling block with two convs and skip.
                One of convs is downsampling.
        """
        filters = sefl.get_shape(inputs)[-1] * 2
        output = conv2d_block(inputs, filters // 2, kernel_size, 'cna', is_training=self.is_training)
        output = conv2d_block(output, filters, kernel_size, 'ca', strides=(2, 2), is_training=self.is_training)
        return output + conv2d_block(inputs, filters, (1, 1), 'c', strides=(2, 2), is_training=self.is_training)

    def _build(self, *args, **kwargs):
        """ Build the model itself.
        """
        # fetch model params
        pic_shape = selg.get_from_config('pic_shape', (32, 32))
        n_channels = self.get_from_config('n_channels', 3)
        n_classes = self.get_from_config('n_classes', 10)

        # set placeholders
        inputs = tf.placeholder(tf.float32, shape=(-1, ) + pic_shape + (n_channels, ), name='images')
        targets = tf.placeholder(tf.float32, shape=(-1, ) + (n_classes, ), name='targets')

        # resnet
        net = self.identity_block(inputs)
        net = self.halfing_block(net)
        net = self.identity_block(net)
        net = self.halfing_block(net)
        net = self.halfing_block(net)
        net = flatten(net)
        net = tf.layers.dense(net, n_classes, name='predictions')

# set up training pipeline
train_ppl = (Pipeline()
             .init_model('static', SmallResNet, 'smallnet',
                         dict(pic_shape=(30, 30), n_channels=2, n_classes=11))
             .train_model('smallnet', feed_dict={'images': B('images'), 'targets': B('labels')})
             )
