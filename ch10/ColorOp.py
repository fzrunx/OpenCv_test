# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:45:22 2025

@author: user
"""

import cv2
import numpy as np

def color_op():
    img = cv2.imread("butterfly.jpg")
    if img is None:
        print("Image load failed!")
        return

    b1, g1, r1 = img[0,0]          # 픽셀 접근
    b2, g2, r2 = img[0,0].tolist() # ptr 접근과 유사

def color_inverse():
    src = cv2.imread("butterfly.jpg")
    if src is None:
        print("Image load failed!")
        return

    dst = 255 - src  # 한 줄로 색상 반전

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_grayscale():
    src = cv2.imread("butterfly.jpg")
    if src is None:
        print("Image load failed!")
        return

    dst = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def color_split():
    src = cv2.imread("candies.png")
    if src is None:
        print("Image load failed!")
        return

    b, g, r = cv2.split(src)

    cv2.imshow("src", src)
    cv2.imshow("B_plane", b)
    cv2.imshow("G_plane", g)
    cv2.imshow("R_plane", r)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    color_op()
    color_inverse()
    color_grayscale()
    color_split()