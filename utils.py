import pickle
import os
import PIL.Image
import numpy as np

def read_in_dict(src, components):
    """ Read all cifar-10 dataset in a dict.
    """
    res = dict()
    for comp in components:
        with open(os.path.join(src, comp + '.pkl'), 'rb') as file:
            data = pickle.load(file)

        res.update({comp:data})

    return res


def read_labs(path='D:/Work/OpenData/cifar-10-batches-py/batches.meta'):
    """ Read labels for cifar-10 dataset.
    """
    with open(path, 'rb') as file:
        labs = pickle.load(file)

    return labs['label_names']


def get_pics_preds(model, batch, size=(256, 256)):
    """ Get pics and predictions.

    Return:
        (pics, preds)
    """
    p, t = model.predict(('predictions', 'targets'),
                         feed_dict={'images': batch.images, 'targets': batch.labels})
    pics = [PIL.Image.fromarray(batch.get(ix, 'images'), mode='RGB').resize(size=size, resample=PIL.Image.LANCZOS)
            for ix in batch.indices]
    return softmax(p), t, pics

def softmax(arr):
    """ Numpy softmax.
    """
    return np.exp(arr) / np.reshape(np.sum(np.exp(arr), axis=1), (-1, 1))

def get_sorted_ser(item):
    """ Get probabilities of predicted labels for an item with probs in ascending order,
    """
    print('True: ', read_labs()[np.argmax(item[1])])
    return(pd.Series(dict(zip(read_labs(), item[0].round(2))))).sort_values(ascending=False)
