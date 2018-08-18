import numpy as np
import numba as nb

__all__ = ['filter']

class filter():
    def __init__(self, kernel, padding = False, pad_option = 'constant'):
        assert kernel.shape[0]%2 == 1 and kernel.shape[1]%2 == 1, 'only support even size mask/kernel.'
        self.kernel = np.flip(np.flip(kernel ,0), 1)
        self.padding = padding
        self.pad_option = pad_option

    def convolve(self, img):
        if self.padding:
            img = self._padding(img, (self.kernel.shape[0]//2, self.kernel.shape[1]//2), self.pad_option)
        if len(img.shape) == 3:
            img = self._rgb_convolution(img)
        else:
            img = self._convolution(img)
        return img

    @nb.jit
    def _rgb_convolution(self, img):
        kernel_height = self.kernel.shape[0] // 2
        kernel_width = self.kernel.shape[1] // 2
        out_put_img = np.zeros((img.shape[0] - 2 * kernel_height, img.shape[1] - 2 * kernel_width, 3))

        for row in range(out_put_img.shape[0]):
            for col in range(out_put_img.shape[1]):
                for color in range(3):
                    out_put_img[row, col, color] = (img[row: row + 2 * kernel_height + 1,
                                                    col: col + 2 * kernel_width + 1, color] * self.kernel).sum()
        return out_put_img.astype(int)

    @nb.jit
    def _convolution(self, img):
        kernel_height = self.kernel.shape[0]//2
        kernel_width = self.kernel.shape[1]//2
        out_put_img = np.zeros((img.shape[0] - 2 * kernel_height, img.shape[1] - 2*kernel_width))
        for row in range(out_put_img.shape[0]):
            for col in range(out_put_img.shape[1]):
                out_put_img[row][col] = (img[row: row + 2 * kernel_height + 1,
                                         col: col +2 * kernel_width + 1] * self.kernel).sum()
        return out_put_img.astype(int)

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
        if len(mask_size) == 3:
            padded = np.pad(img, mask_size, option)
        else:
            padded =  np.pad(img, ((mask_size[0],mask_size[0]),(mask_size[1],mask_size[1]),(0,0)), option)
        return padded

