import numba as nb
import numpy as np

class TD_k_means_thresholding():
    def __init__(self, pad = 0,initial = [0, 255], maxiter = 10000):
        self.initial = initial
        self.maxiter = maxiter
        self.img = None
        self.filterID = 0
        self.pad = pad

    def input_img(self, img):
        self.img = img

    def _k_means_clustering(self, data):
        cluster_mean1, cluster_mean2 = self.initial[0], self.initial[1]
        prev1, prev2 = -1, -1
        for i in range(self.maxiter):
            if prev1 == cluster_mean1 and prev2 == cluster_mean2:
                break
            cluster_sum_1 = 0
            cluster_sum_2 = 0
            num_1 = 0
            num_2 = 0
            prev1, prev2 = cluster_mean1, cluster_mean2
            cluster_mean1, cluster_mean2 = self._assign_pixels_binary(data, cluster_sum_1,
                                                                     cluster_sum_2, num_1, num_2,
                                                                     cluster_mean1, cluster_mean2)
        return [cluster_mean1, cluster_mean2]

    @nb.jit
    def _assign_pixels_binary(self, data, cluster_sum_1, cluster_sum_2, num_1, num_2, cluster_mean1, cluster_mean2):
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                pixel = data[x][y]
                if abs(cluster_mean1 - pixel) < abs(cluster_mean2 - pixel):
                    cluster_sum_1 += pixel
                    num_1 += 1
                else:
                    cluster_sum_2 += pixel
                    num_2 += 1
        return cluster_sum_1 / num_1, cluster_sum_2 / num_2

    @nb.jit
    def binary_output(self):
        assert self.img is not None, 'Please call in-class function input_img first to save input image.'
        data = self.img
        threshold_origin = self._k_means_clustering(data)
        binary_img = np.zeros(data.shape)
        avg_img = self._get_avg_img(data)
        threshold_avg = self._k_means_clustering(avg_img)
        threshold = (threshold_origin[0] + threshold_origin[1])/2 + (abs(threshold_avg[0] - threshold_origin[0]) + abs(threshold_avg[1] - threshold_origin[1]))/2
        for x in range(binary_img.shape[0]):
            for y in range(binary_img.shape[1]):
                if data[x][y] >= threshold:
                    binary_img[x][y] = 255
        return np.uint8(binary_img.astype(int))

    def _get_avg_img(self, img):
        avg_img = np.zeros(img.shape)
        n = self.pad
        n_square = n**2
        pad_img = np.pad(img, (n//2, n//2), 'edge')
        for i in range(avg_img.shape[0]):
            for j in range(avg_img.shape[1]):
                avg_img[i][j] = (pad_img[i: i + n,
                                         j: j + n].sum())/n_square
        return avg_img