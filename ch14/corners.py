# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:16:15 2025

@author: user
"""

import cv2
import numpy as np

def corner_harris():
    src = cv2.imread("building.jpg", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    harris = cv2.cornerHarris(src, 3, 3, 0.04)
    harris_norm = cv2.normalize(harris, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    for j in range(1, harris.shape[0] - 1):
        for i in range(1, harris.shape[1] - 1):
            if harris_norm[j, i] > 120:
                if (harris[j, i] > harris[j-1, i] and
                    harris[j, i] > harris[j+1, i] and
                    harris[j, i] > harris[j, i-1] and
                    harris[j, i] > harris[j, i+1]):
                    cv2.circle(dst, (i, j), 5, (0, 0, 255), 2)

    cv2.imshow("src", src)
    cv2.imshow("harris_norm", harris_norm)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def corner_fast():
    src = cv2.imread("building.jpg", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    fast = cv2.FastFeatureDetector_create(threshold=60, nonmaxSuppression=True)
    keypoints = fast.detect(src, None)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    for kp in keypoints:
        cv2.circle(dst, (int(kp.pt[0]), int(kp.pt[1])), 5, (0, 0, 255), 2)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    corner_harris()
    corner_fast()