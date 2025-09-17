# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:46:32 2025

@author: user
"""

import cv2
import numpy as np

# 초기 Hue 값
lower_hue = 40
upper_hue = 80

# 이미지 로드
src = cv2.imread("candies.png")
if src is None:
    print("Image load failed!")
    exit()

# BGR → HSV 변환
src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# 트랙바 콜백 함수
def on_hue_changed(x):
    lowerb = np.array([cv2.getTrackbarPos("Lower Hue", "mask"), 100, 0])
    upperb = np.array([cv2.getTrackbarPos("Upper Hue", "mask"), 255, 255])
    mask = cv2.inRange(src_hsv, lowerb, upperb)
    cv2.imshow("mask", mask)

# 창 및 트랙바 생성
cv2.imshow("src", src)
cv2.namedWindow("mask")
cv2.createTrackbar("Lower Hue", "mask", lower_hue, 179, on_hue_changed)
cv2.createTrackbar("Upper Hue", "mask", upper_hue, 179, on_hue_changed)

# 초기 마스크 표시
on_hue_changed(0)

cv2.waitKey(0)
cv2.destroyAllWindows()