# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:57:24 2025

@author: user
"""

import cv2

# 이미지 로드
src = cv2.imread("neutrophils.png", cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed!")
    exit()

# 트랙바 콜백
def on_threshold(pos):
    _, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow("dst", dst)

cv2.imshow("src", src)
cv2.namedWindow("dst")
cv2.createTrackbar("Threshold", "dst", 128, 255, on_threshold)

# 초기 화면 표시
on_threshold(128)

cv2.waitKey(0)
cv2.destroyAllWindows()