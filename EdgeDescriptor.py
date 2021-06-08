import cv2
import numpy as np
from math import sqrt


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

class Edge:

    def histogram(self, image, stride=(2, 2), normalize=True):

        image = image.astype('uint8')
        hist = self._conv(image, stride=stride, kernels=edge_kernels)
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




#image = cv2.imread("queries/art294.jpg")
#test = Edge()
#value = test.histogram(image)
#print(value)