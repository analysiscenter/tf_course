""" Small 2d-resnet. """

from cifar_batch import Cifar10Batch
import tensorflow as tf

from dataset.dataset.models.tf.layers import conv_block
from dataset.dataset.models.tf import TFModel

# global constants for shape of pics
PIC_SHAPE = (32, 32)
N_CHANNELS = 3
N_CLASSES = 10

class TestModel(TFModel):
    """ Test-model for figuring out configs in new version of TFModel
    """
    @classmethod
    def default_config(cls):
        config = TFModel.default_config()
        config['body'].update(layout='cap cnap ca', filters=[16, 32, 64], kernel_size=[5, 3, 3])  # importantly, kernel_size here is list
                                                                                                  # when list, different kernels are used for different convs
        config['head'].update(layout='Pf', units=N_CLASSES)
        config['inputs'].update(images=dict(shape=(*PIC_SHAPE, N_CHANNELS)),                      # note that shape is specified without batch-size-axis (no None)
                                labels=dict(shape=(N_CLASSES, ), name='targets'))
        config['input_block'].update(inputs='images')
        config['output'].update(ops='accuracy')

        config['loss'] = 'ce'
        config['optimizer'] = 'Adam'
        return config


    @classmethod
    def body(cls, inputs, name='body', **kwargs):
        print(kwargs)
        kwargs = cls.fill_params('body', **kwargs) # the line isn't needed here.
                                                   # It means "take default_config['body'] and merge it with body-args that came from kwargs"
                                                   # is_training is put into kwargs from default_config.
        print('filled args in body: ', kwargs)
        output = conv_block(inputs, **kwargs)
        return output


class SmallResNet(TFModel):
    """ Model-class implementing small resnet.
    """

    def identity_block(self, inputs, kernel_size=(3, 3)):
        """ Build identity block with two convs and skip.
        """
        filters = self.get_shape(inputs)[-1]
        output = conv2d_block(inputs, filters, kernel_size, 'ca', is_training=self.is_training)
        output = conv2d_block(output, filters, kernel_size, 'cna', is_training=self.is_training)
        return output + inputs

    def halfing_block(self, inputs, kernel_size=(3, 3)):
        """ Build downsampling block with two convs and skip.
                One of convs is downsampling.
        """
        filters = self.get_shape(inputs)[-1] * 2
        output = conv2d_block(inputs, filters // 2, kernel_size, 'ca', is_training=self.is_training)
        output = conv2d_block(output, filters, kernel_size, 'cna', strides=(2, 2), is_training=self.is_training)
        return output + conv2d_block(inputs, filters, (1, 1), 'c', strides=(2, 2), is_training=self.is_training)

    def _build(self, *args, **kwargs):
        """ Build the model itself.
        """
        # fetch model params
        pic_shape = self.get_from_config('pic_shape', (32, 32))
        n_channels = self.get_from_config('n_channels', 3)
        n_classes = self.get_from_config('n_classes', 10)

        # set placeholders
        inputs = tf.placeholder(tf.float32, shape=(None, ) + pic_shape + (n_channels, ), name='images')
        targets = tf.placeholder(tf.float32, shape=(None, ) + (n_classes, ), name='targets')

        # net
        net = conv2d_block(inputs, 64, (7, 7), 'cnap', is_training=self.is_training)
        net = self.identity_block(net)
        net = self.identity_block(net)
        net = self.halfing_block(net)
        net = self.identity_block(net)
        net = self.identity_block(net)
        net = self.halfing_block(net)
        net = self.identity_block(net)
        net = self.identity_block(net)
        net = self.halfing_block(net)
        net = iflatten(net)
        net = tf.layers.dense(net, 256)
        net = tf.layers.dense(net, n_classes)
        net = tf.identity(net, name='predictions')

        # stats
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(tf.nn.softmax(net), axis=1),
                                                   tf.argmax(targets, axis=1)), tf.float32))
        accuracy = tf.identity(accuracy, name='accuracy')
