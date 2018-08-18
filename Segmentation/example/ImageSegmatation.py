from Segmentation import k_means_thresholding
from Segmentation import filtered_k_means_thresholding
from Segmentation import k_means_thresholding2D
import numpy as np
import cv2

img = cv2.imread('./TestImage/input.jpg', 0)
km = k_means_thresholding.k_means_thresholding(initial=[0, 255])
img1 = km.binary_output(img)
cv2.imwrite('./TestImage/output_kmeans.jpg', img1)


km = filtered_k_means_thresholding.filtered_k_means_thresholding(initial=[0, 255])
km.input_img(img)
sharper5 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
km.filter_img(sharper5)
km.filter_img(sharper5)
km.blur_img('gaussian', 3)
img2 = km.binary_output()
img2 = cv2.cv2.GaussianBlur(img2, (3, 3), 10)
cv2.imwrite('./TestImage/output_kmeans_filtered.jpg', img2)


km = k_means_thresholding2D.TD_k_means_thresholding(25)
km.input_img(img)
img3 = km.binary_output()
cv2.imwrite('./TestImage/output_kmeans_2D.jpg', img3)