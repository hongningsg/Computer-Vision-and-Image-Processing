from Segmentation import k_means_thresholding
from Segmentation import filtered_k_means_thresholding
from Segmentation import k_means_thresholding2D
import numpy as np
import cv2

'''
Use KMeans algorithm to segment frog egg image
'''
img = cv2.imread('./TestImage/input.jpg', 0)
km = k_means_thresholding.k_means_thresholding(initial=[0, 255])
img1 = km.binary_output(img)
cv2.imwrite('./TestImage/output_kmeans.jpg', img1)

'''
Use KMeans algorithm to segment frog egg image.
Before K-Means thresholding:
Applied sharpen filter
0  -1  0
-1  5  -1
0  -1  0
twice to make edges more obviously, then applied a Gaussian Blur
 with mask size of 3x3 and standard deviation of 3 to remove some noise.

After K-Means thresholding:
Applied a Gaussian Blur with mask size of 3x3 and standard deviation of 10 to make binary image looks more smooth.

'''
km = filtered_k_means_thresholding.filtered_k_means_thresholding(initial=[0, 255])
km.input_img(img)
sharper5 = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
km.filter_img(sharper5)
km.filter_img(sharper5)
km.blur_img('gaussian', 3)
img2 = km.binary_output()
img2 = cv2.cv2.GaussianBlur(img2, (3, 3), 10)
cv2.imwrite('./TestImage/output_kmeans_filtered.jpg', img2)

'''
Applied second channel with averaging smoothing image to find out the threshold
'''
km = k_means_thresholding2D.TD_k_means_thresholding(25)
km.input_img(img)
img3 = km.binary_output()
cv2.imwrite('./TestImage/output_kmeans_2D.jpg', img3)