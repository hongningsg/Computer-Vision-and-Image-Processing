'''
Important Note: This Algorithm patent owned by David G.Lowe (The University of British Columbia) and not included in
free version of open-cv. To import SIFT, please do as follow:
# $ python -m pip install opencv-contrib-python

To check installation success:
# >>> import cv2
# >>> surf = cv2.xfeatures2d.SURF_create()
'''

import numpy as np
import cv2
import sys

def getsize(img):
    h, w = img.shape[:2]
    return w, h

class image_feature_detector(object):
    SIFT = 0
    SURF = 1

    def __init__(self, feat_type, params = None):
        self.detector, self.norm = self.features_detector(feat_type=feat_type, params = params)

    def features_detector(self, feat_type = SIFT, params = None):
        if feat_type == self.SIFT:
            if params is None:
                nfeatures = 0
                nOctaveLayers = 3
                contrastThreshold = 0.04
                edgeThreshold=10
                sigma=1.6
            else:
                nfeatures = params["nfeatures"]
                nOctaveLayers = params["nOctaveLayers"]
                contrastThreshold = params["contrastThreshold"]
                edgeThreshold = params["edgeThreshold"]
                sigma = params["sigma"]

            detector = cv2. xfeatures2d.SIFT_create(nfeatures=nfeatures, nOctaveLayers=nOctaveLayers, contrastThreshold=contrastThreshold,
                                                        edgeThreshold=edgeThreshold, sigma=sigma)
            norm = cv2.NORM_L2
        elif feat_type == self.SURF:
            if params is None:
                hessianThreshold = 3000
                nOctaves = 1
                nOctaveLayers = 1
                upright = True
                extended = False
            else:
                hessianThreshold = params["hessianThreshold"]
                nOctaves = params["nOctaves"]
                nOctaveLayers = params["nOctaveLayers"]
                upright = params["upright"]
                extended = params["extended"]

            detector = cv2.xfeatures2d.SURF_create(hessianThreshold=hessianThreshold,
                                                       nOctaves = nOctaves, nOctaveLayers = nOctaveLayers,
                                                        upright = upright, extended = extended)
            norm = cv2.NORM_L2

        return detector, norm

def sift(img):
    sift = cv2.xfeatures2d.SIFT_create(nfeatures=0,nOctaveLayers=3, contrastThreshold=0.04,
                                                   edgeThreshold=10, sigma=1.6)
    kp = sift.detect(img,None)
    img = cv2.drawKeypoints(img, kp, outImage=np.array([]), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return img

if __name__ == "__main__":
    sift_detect = image_feature_detector(feat_type=0)
    surf_detect = image_feature_detector(feat_type=1)
    # Implement yourself
    ## Refer to opencv documentation,
    ## use SIFT or SURF on the test image and show detection result.
    ### 1. Read Image
    ### 2. Convert the image to greyscale
    ### 3. Initialize an SIFT detector
    ### 4. Feature detection and visualize the detected features
    img1 = cv2.imread('./example/Image/parrot.jpg', 0)
    img2 = cv2.imread('./example/Image/bit.jpg', 0)
    sift_img1 = sift_detect.detector.detect(img1, None)
    out_img1 = cv2.drawKeypoints(img1, sift_img1, img1)
    sift_img2 = sift_detect.detector.detect(img2, None)
    out_img2 = cv2.drawKeypoints(img2, sift_img2, img2)
    cv2.imwrite('./example/Image/sift_parrot.jpg', out_img1)
    cv2.imwrite('./example/Image/sift_bit.jpg', out_img2)

    surf_img1 = surf_detect.detector.detect(img1, None)
    out_img1 = cv2.drawKeypoints(img1, surf_img1, img1)
    cv2.imwrite('./example/Image/surf_parrot.jpg', out_img1)
    surf_img2 = surf_detect.detector.detect(img2, None)
    out_img2 = cv2.drawKeypoints(img2, surf_img2, img2)
    cv2.imwrite('./example/Image/surf_bit.jpg', out_img2)
