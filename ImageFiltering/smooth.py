import numpy as np

__all__ = ['smooth']

class smooth():
    def __init__(self, kernel, padding = False, pad_option = 'edge'):
        assert kernel.shape[0]%2 == 1 and kernel.shape[1]%2 == 1, 'only support even size mask/kernel.'
        self.kernel = kernel
        self.padding = padding
        self.pad_option = pad_option

    def smoothing(self, img):
        if self.padding:
            img = self._padding(img, (self.kernel.shape[0]//2, self.kernel.shape[1]//2), self.pad_option)
        if len(img.shape) == 3:
            img = self._rgb_convolution(img)
        else:
            img = self._convolution(img)
        return img

    def _rgb_convolution(self, img):
        kernel_height = self.kernel.shape[0] // 2
        kernel_width = self.kernel.shape[1] // 2
        out_put_img = np.zeros((img.shape[0] - kernel_height, img.shape[1] - kernel_width, 3))

        for color in range(img.shape[2]):
            for row in range(out_put_img.shape[0]):
                for col in range(out_put_img.shape[1]):
                    out_put_img[row][col][color] = (img[row - kernel_width: row + kernel_width,
                                                    col - kernel_width: col + kernel_width][color] * self.kernel).sum()
        return out_put_img

    def _convolution(self, img):
        kernel_height = self.kernel.shape[0]//2
        kernel_width = self.kernel.shape[1]//2
        out_put_img = np.zeros((img.shape[0] - 2*kernel_height, img.shape[1] - 2*kernel_width))
        for row in range(out_put_img.shape[0]):
            for col in range(out_put_img.shape[1]):
                out_put_img[row][col] = (img[row: row + 2*kernel_width+1,
                                         col: col +2*kernel_width+1] * self.kernel).sum()
        return out_put_img

    def _mean_filter(self):
        pass

    def _padding(self, img, mask_size, option):
        methods = ['edge',
                   'mean',
                   'wrap',
                   'constant',
                   'median',
                   'minimum',
                   'reflect',
                   'symmetric']
        assert option in methods, 'Padding option not support'
        return np.pad(img, mask_size, option)



if __name__ == "__main__":
    import cv2
    TestImg = cv2.imread('../Test_Image/cute-red-panda-01.jpg',0)
    b = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],[1, 2, 3, 4, 5], [6, 7, 8, 9, 10],[1, 2, 3, 4, 5]])
    a = np.array([[1, 1, 1], [2, 2, 2], [-1, -1, -1]])
    test = np.array([[3,0,1,5,0],[4,3,0,3,0],[2,4,1,0,6],[3,0,1,5,0]])
    Imgfilter = smooth(a, padding=True, pad_option='constant')
    print(Imgfilter.smoothing(test))
    TestImg = Imgfilter.smoothing(TestImg)
    cv2.imwrite('../Test_Output/cute-red-panda-01.jpg', TestImg)
