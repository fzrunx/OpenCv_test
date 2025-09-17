# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:03:16 2025

@author: user
"""

import cv2
import sys

# --- sys.argv 대신 직접 이미지 경로 지정 ---
imgs = [cv2.imread("img1.jpg"), cv2.imread("img2.jpg"),cv2.imread("img3.jpg")]  # 실제 파일명으로 바꾸기

if any(img is None for img in imgs):
    print("Image load failed!")
    sys.exit()

stitcher = cv2.Stitcher_create()
status, dst = stitcher.stitch(imgs)

if status != cv2.Stitcher_OK:
    print("Error on stitching!")
    sys.exit()

cv2.imwrite("result.jpg", dst)
cv2.imshow("stitched", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()