import numba as nb
import numpy as np
import cv2

class filtered_k_means_thresholding:
    def __init__(self, initial = [], maxiter = 10000):
        self.initial = initial
        self.maxiter = maxiter
        self.img = None

    def input_img(self, img):
        self.img = img

    def filter_img(self, kernel):
        self.img = cv2.filter2D(self.img, -1, kernel)

    def blur_img(self, blur, square):
        if blur == 'avg':
            self.img = cv2.blur(self.img,(square,square))
        elif blur == 'gaussian':
            self.img = cv2.GaussianBlur(self.img, (square, square), 3)
        elif blur == 'med':
            self.img = cv2.medianBlur(self.img, square)
        elif blur == 'bilateral':
            self.img = cv2.bilateralFilter(self.img, square, 80, 80)
        else:
            print('Blur not recognised, no blur applied')

    def k_means_clustering(self):
        assert self.img is not None, 'Please call in-class function input_img first to save input image.'
        data = self.img
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
            cluster_mean1, cluster_mean2 = self.assign_pixels_binary(data, cluster_sum_1,
                                                                     cluster_sum_2, num_1, num_2,
                                                                     cluster_mean1, cluster_mean2)
        return (cluster_mean1 + cluster_mean2)/2

    @nb.jit
    def assign_pixels_binary(self, data, cluster_sum_1, cluster_sum_2, num_1, num_2, cluster_mean1, cluster_mean2):
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
        threshold = self.k_means_clustering()
        binary_img = np.zeros(data.shape)
        for x in range(binary_img.shape[0]):
            for y in range(binary_img.shape[1]):
                if data[x][y] >= threshold:
                    binary_img[x][y] = 255
        return np.uint8(binary_img.astype(int))