from Partitioning import Partition
import cv2

'''
This is an example of using Two-Pass Algorithm to count groups in an image.
With 4 neighbors rules, input should be a binary image(i,e,. only will be 
regarded as black and white), output is number of groups in input image and
a colored image that indicates groups intuitively.
'''
img = cv2.imread('./TestImage/input.jpg',0)
par = Partition.Partition(threshold=0).separate(img)
cv2.imwrite('./TestImage/output.jpg',par)