import numpy as np
import numba as nb
import sys

__all__ = ['Partition']

class Partition():
    def __init__(self, rule = 4, threshold = 20, background = 132):
        assert rule==4 or rule == 8, 'Only support 4 neighbors or 8 neighbors rules.'
        self.rule = rule
        self.threshold = threshold
        self.background = background
        self._groups = {}

    def separate(self, img):
        visited = np.zeros(img.shape, dtype=int)
        if self.rule == 4:
            visited, union = self._first_pass4(img, visited)
        else:
            visited, union = self._first_pass8(img, visited)
        visited = self._second_pass(visited, union)
        visited = self._size_filter(visited)
        num_of_g = self._num_of_group()
        sys.stdout.write("Number of groups:" + str(num_of_g))
        return self._color_photo(num_of_g, visited)

    @nb.jit
    def _first_pass4(self, img, visit):
        label = 1
        union = dict()
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y] < self.background:
                    neighbors = []
                    if x > 0 and visit[x-1][y] != 0:
                        neighbors.append(visit[x-1][y])
                    if x < visit.shape[0] - 1 and visit[x+1][y] != 0:
                        neighbors.append(visit[x+1][y])
                    if y > 0 and visit[x][y-1] != 0:
                        neighbors.append(visit[x][y-1])
                    if y < visit.shape[1] - 1 and visit[x][y+1] != 0:
                        neighbors.append(visit[x][y+1])

                    if len(neighbors) == 0:
                        union[label] = set([label])
                        visit[x][y] = label
                        label += 1
                    else:
                        L = min(neighbors)
                        visit[x][y] = L
                        for n in neighbors:
                            union[n] = union[n]|set(neighbors)
        return visit, union

    @nb.jit
    def _first_pass8(self, img, visit):
        label = 1
        union = dict()
        for x in range(img.shape[0]):
            for y in range(img.shape[1]):
                if img[x][y] < self.background:
                    neighbors = []
                    if x > 0 and visit[x-1][y] != 0:
                        neighbors.append(visit[x-1][y])
                    if x < visit.shape[0] - 1 and visit[x+1][y] != 0:
                        neighbors.append(visit[x+1][y])
                    if y > 0 and visit[x][y-1] != 0:
                        neighbors.append(visit[x][y-1])
                    if y < visit.shape[1] - 1 and visit[x][y+1] != 0:
                        neighbors.append(visit[x][y+1])
                    if y > 0 and x > 0 and visit[x-1][y-1] != 0:
                        neighbors.append(visit[x-1][y-1])
                    if y > 0 and x < visit.shape[0] - 1 and visit[x+1][y-1] != 0:
                        neighbors.append(visit[x+1][y-1])
                    if y < visit.shape[1] - 1 and x > 0 and visit[x-1][y+1] != 0:
                        neighbors.append(visit[x-1][y+1])
                    if y < visit.shape[1] - 1 and x < visit.shape[0] - 1 and visit[x+1][y+1] != 0:
                        neighbors.append(visit[x+1][y+1])
                    if len(neighbors) == 0:
                        union[label] = set([label])
                        visit[x][y] = label
                        label += 1
                    else:
                        L = min(neighbors)
                        visit[x][y] = L
                        for n in neighbors:
                            union[n] = union[n]|set(neighbors)
        return visit, union

    def _second_pass(self, visit, union):
        self._groups = {key:0 for key in union}
        for x in range(visit.shape[0]):
            for y in range(visit.shape[1]):
                if visit[x][y] != 0:
                    visit[x][y] = self._uni_find(union, visit[x][y])
                    self._groups[visit[x][y]] += 1
        return visit

    def _uni_find(self, union, key):
        if min(union[key]) == key:
            return key
        return self._uni_find(union, min(union[key]))

    @nb.jit
    def _size_filter(self, visit):
        for x in range(visit.shape[0]):
            for y in range(visit.shape[1]):
                if visit[x][y] != 0 and self._groups[visit[x][y]] <= self.threshold:
                    visit[x][y] = 0
        return visit

    @nb.jit
    def _color_photo(self, num_of_color, visit):
        out_img = np.zeros((visit.shape[0], visit.shape[1], 3))
        total_color = 16777116
        color_inc = total_color//num_of_color
        color = 100
        colored = {}
        for x in range(visit.shape[0]):
            for y in range(visit.shape[1]):
                if visit[x][y] != 0:
                    if self.filtered_group[visit[x][y]] == 0:
                        curr_color = self._Dto256(color)
                        colored[visit[x][y]] = curr_color
                        self.filtered_group[visit[x][y]] = curr_color
                        out_img[x][y] = curr_color
                        color += color_inc
                    else:
                        out_img[x][y] = self.filtered_group[visit[x][y]]
                else:
                    out_img[x][y] = [255, 255, 255]
        return out_img

    def _Dto256(self, color):
        R = color % 256
        G = (color//256)%256
        B = (color//65536)%256
        return [R, G, B]

    @nb.jit
    def _num_of_group(self):
        count = 0
        self.filtered_group = dict()
        for group in self._groups:
            if self._groups[group] > self.threshold:
                self.filtered_group[group] = 0
                count += 1
        return count

    def get_group(self):
        return self._groups