from __future__ import print_function

from evaluate import evaluate_class
from DB import Database

from six.moves import cPickle
import numpy as np
import scipy.misc
from math import sqrt
import os

stride = (1, 1)
n_slice = 10
h_type = 'region'
d_type = 'cosine'

depth = 5

edge_kernels = np.array([
    [
        # vertical
        [1, -1],
        [1, -1]
    ],
    [
        # horizontal
        [1, 1],
        [-1, -1]
    ],
    [
        # 45 diagonal
        [sqrt(2), 0],
        [0, -sqrt(2)]
    ],
    [
        # 135 diagnol
        [0, sqrt(2)],
        [-sqrt(2), 0]
    ],
    [
        # non-directional
        [2, -2],
        [-2, 2]
    ]
])

# cache dir
cache_dir = 'cache'
if not os.path.exists(cache_dir):
    os.makedirs(cache_dir)


class Edge(object):

    def histogram(self, input, stride=(2, 2), normalize=True):
        ''' count img histogram

          arguments
            input    : a path to a image or a numpy.ndarray
            stride   : stride of edge kernel
            type     : 'global' means count the histogram for whole image
                       'region' means count the histogram for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output histogram

          return
            type == 'global'
              a numpy array with size len(edge_kernels)
            type == 'region'
              a numpy array with size len(edge_kernels) * n_slice * n_slice
        '''
        if isinstance(input, np.ndarray):  # examinate input type
            img = input.copy()
        else:
            img = scipy.misc.imread(input, mode='RGB')



        hist = self._conv(img, stride=stride, kernels=edge_kernels)



        if normalize:
            hist /= np.sum(hist)

        return hist.flatten()

    def _conv(self, img, stride, kernels, normalize=True):
        H, W, C = img.shape
        conv_kernels = np.expand_dims(kernels, axis=3)
        conv_kernels = np.tile(conv_kernels, (1, 1, 1, C))
        assert list(conv_kernels.shape) == list(kernels.shape) + [C]  # check kernels size

        sh, sw = stride
        kn, kh, kw, kc = conv_kernels.shape

        hh = int((H - kh) / sh + 1)
        ww = int((W - kw) / sw + 1)

        hist = np.zeros(kn)

        for idx, k in enumerate(conv_kernels):
            for h in range(hh):
                hs = int(h * sh)
                he = int(h * sh + kh)
                for w in range(ww):
                    ws = w * sw
                    we = w * sw + kw
                    hist[idx] += np.sum(img[hs:he, ws:we] * k)  # element-wise product

        if normalize:
            hist /= np.sum(hist)

        return hist

    def make_samples(self, db, verbose=True):

        sample_cache = "edge-{}-stride{}".format(h_type, stride)


        try:
            samples = cPickle.load(open(os.path.join(cache_dir, sample_cache), "rb", True))
            for sample in samples:
                sample['hist'] /= np.sum(sample['hist'])  # normalize
            if verbose:
                print("Using cache..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))
        except:
            if verbose:
                print("Counting histogram..., config=%s, distance=%s, depth=%s" % (sample_cache, d_type, depth))

            samples = []
            data = db.get_data()
            for d in data.itertuples():
                d_img, d_cls = getattr(d, "img"), getattr(d, "cls")
                d_hist = self.histogram(d_img, type=h_type, n_slice=n_slice)
                samples.append({
                    'img': d_img,
                    'cls': d_cls,
                    'hist': d_hist
                })
            cPickle.dump(samples, open(os.path.join(cache_dir, sample_cache), "wb", True))

        return samples


