# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:43:16 2025

@author: user
"""

import cv2
import numpy as np

# Reference image histogram
ref = cv2.imread("ref.png")
mask = cv2.imread("mask.bmp", cv2.IMREAD_GRAYSCALE)
ref_ycrcb = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

channels = [1, 2]  # Cr, Cb
histSize = [128, 128]
ranges = [0, 256, 0, 256]

hist = cv2.calcHist([ref_ycrcb], channels, mask, histSize, ranges)

# Backprojection on input image
src = cv2.imread("kids.png")
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, scale=1)

cv2.imshow("src", src)
cv2.imshow("backproj", backproj)
cv2.waitKey(0)
cv2.destroyAllWindows()