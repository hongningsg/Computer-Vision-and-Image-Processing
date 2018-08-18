import numpy as np
import numba as nb
import random
import time

class k_means_thresholding():
    def __init__(self, k = 2, initial = [], maxiter = 10000):
        if len(initial) == 0:
            initial = [random.randint(0, 255) for _ in range(k)]
        assert len(initial) == k, "Number of classes(k) is not the same as number of initial thresholds."
        self.k = k
        self.initial = initial
        self.maxiter = maxiter

    def k_means_clustering(self, data):
        k = self.k
        cluster_mean = self.initial
        prev = [-1] * k
        for i in range(self.maxiter):
            if prev == cluster_mean:
                break
            cluster = {j: [0, 0] for j in range(k)}
            cluster = self.assign_pixels(data, cluster, cluster_mean)
            prev = cluster_mean.copy()
            cluster_mean = self.new_means(cluster)
        return sum(cluster_mean)/len(cluster_mean)

    def two_means_clustering(self, data):
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
    def assign_pixels(self, data, cluster, cluster_mean):
        for x in range(data.shape[0]):
            for y in range(data.shape[1]):
                closest = 256
                belonging = -1
                pixel = data[x][y]
                for c in cluster:
                    distance = abs(cluster_mean[c] - pixel)
                    if distance < closest:
                        closest = distance
                        belonging = c
                cluster[belonging][0] += pixel
                cluster[belonging][1] += 1
        return cluster

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
        return cluster_sum_1/num_1, cluster_sum_2/num_2

    @nb.jit
    def new_means(self, cluster):
        means = list()
        for key in cluster:
            if cluster[key][1] == 0:
                cluster[key][1] = 1
            means.append(cluster[key][0]/cluster[key][1])
        return means

    @nb.jit
    def binary_output(self, data):
        threshold = self.two_means_clustering(data)
        #threshold = self.k_means_clustering(data)
        binary_img = np.zeros(data.shape)
        for x in range(binary_img.shape[0]):
            for y in range(binary_img.shape[1]):
                if data[x][y] >= threshold:
                    binary_img[x][y] = 255
        return binary_img.astype(int)