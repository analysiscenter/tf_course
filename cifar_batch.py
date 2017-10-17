""" Custom batch class for storing cifar-10 batch and models
"""
import numpy as np
import os
import pickle
from dataset import Batch, action, model

class Cifar10Batch(Batch):
    """ Cifar-10 batch.
    """
    def __init__(self, index, *args, **kwargs):
        """ Init func.

        Args:
            ___

        Return:
            ___
        """
        super().init(index, *args, **kwargs)
        self.images = None
        self.labels = None
        self.picnames = None

    @property
    def components(self):
        """ Components of cifar-10 batch.

        images: ndarray of shape (batch_size, 3, 32, 32), containig
            cifar-10 images in RGB-mode.

        labels: ndarray of shape (batch_size, 10/-1) containing labels
            of cifar-10 images in one-hot/int format.

        picnames: ndarray of shape (batch_size, ) containing filenames of
            cifar-10 images.
        """
        return 'images', 'labels', 'picnames'

    @staticmethod
    def _adjust_shape(component, name):
        """ Adjust the shape of an array representing a component when loading from
                pickle/memory(ndarray).

        Args:
            component: array containing component's data.
            name: the name of the component.

        Return:
            an array with adjusted shape; component.
        """
        if name == 'images':
            return component.reshape(-1, 3, 32, 32).transpose((0, 2, 3, 1))
        elif name == 'labels':
            one_hot_labels = np.zeros(shape=(len(component), 10))
            flattened = np.reshape(component, -1)
            one_hot_labels[np.arange(len(flattened)), flattened] = 1
            return one_hot_labels
        elif name == 'picnames':
            return np.reshape(component, (-1, 1))


    @action
    def load(self, src, fmt='pkl'):
        """ Load cifar-10 pics.

        Args:
            src: if fmt is 'pkl', then src is assumed to be a path to a folder containing
                pickled files with components ('component.pkl') that should be loaded.
                if fmt is 'ndarray', then src is assumed to be a dict with keys that correspond
                to compoenents to be loaded. In both cases ndarrays are subindexed according
                to indices in batch.
            fmt: format of src. Can be either 'pkl' (pickle) or 'ndarray'.
            nclasses: type of cifar. 10 corresponds to cifar-10.

        Return:
            self.
        """
        if fmt == 'pkl':
            components = set(os.listdir(src)) | set([comp + '.pkl' for comp in self.components])
            for comp in components:
                with open(os.path.join(src, comp), 'rb') as file:
                    component = self._adjust_shape(pickle.load(file)[self.indices], comp.split('.')[0])
                    setattr(self, comp.split('.')[0], component)
        elif fmt == 'ndarray':
            for comp in src:
                setattr(self, comp, self._adjust_shape(src.get(comp)[self.indices], comp))
