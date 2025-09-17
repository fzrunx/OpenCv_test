# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:44:17 2025

@author: user
"""

import cv2

src = cv2.imread("pepper.bmp")
if src is None:
    print("Image load failed!")
    exit()

src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(src_ycrcb)

y_eq = cv2.equalizeHist(y)
dst_ycrcb = cv2.merge([y_eq, cr, cb])
dst = cv2.cvtColor(dst_ycrcb, cv2.COLOR_YCrCb2BGR)

cv2.imshow("src", src)
cv2.imshow("dst", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()