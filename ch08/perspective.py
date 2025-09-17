# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:34:34 2025

@author: user
"""

import cv2
import numpy as np

src = cv2.imread("card.bmp")
if src is None:
    print("Image load failed!")
    exit()

src_pts = []
dst_pts = np.float32([[0,0], [199,0], [199,299], [0,299]])

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(src_pts) < 4:
        src_pts.append([x, y])
        cv2.circle(src, (x, y), 5, (0,0,255), -1)
        cv2.imshow("src", src)

        if len(src_pts) == 4:
            pers = cv2.getPerspectiveTransform(np.float32(src_pts), dst_pts)
            dst = cv2.warpPerspective(src, pers, (200, 300))
            cv2.imshow("dst", dst)

cv2.imshow("src", src)
cv2.setMouseCallback("src", on_mouse)
cv2.waitKey(0)
cv2.destroyAllWindows()