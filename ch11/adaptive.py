# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:58:21 2025

@author: user
"""

import cv2

src = cv2.imread("sudoku.jpg", cv2.IMREAD_GRAYSCALE)
if src is None:
    print("Image load failed!")
    exit()

def on_trackbar(pos):
    bsize = pos
    if bsize % 2 == 0:
        bsize -= 1
    if bsize < 3:
        bsize = 3
    dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                cv2.THRESH_BINARY, bsize, 2)
    cv2.imshow("dst", dst)

cv2.imshow("src", src)
cv2.namedWindow("dst")
cv2.createTrackbar("Block Size", "dst", 11, 200, on_trackbar)

# 초기 화면 표시
on_trackbar(11)

cv2.waitKey(0)
cv2.destroyAllWindows()