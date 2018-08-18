from ImageFiltering import filter
import numpy as np
import cv2

'''
This is an example of using filter

0  -1  0
-1  5  -1
0  -1  0

to sharpen image. 
'''
TestImg = cv2.imread('./TestImage/input.jpg', -1)
a = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
Imgfilter = filter.filter(a, padding=True, pad_option='edge')
TestImg2 = Imgfilter.convolve(TestImg)
cv2.imwrite('./TestImage/output.jpg', TestImg2)
