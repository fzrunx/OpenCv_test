# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:39:10 2025

@author: user
"""

import cv2
import numpy as np

def sobel_derivative():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    mx = np.array([[-0.5, 0, 0.5]], dtype=np.float32)
    my = np.array([[-0.5], [0], [0.5]], dtype=np.float32)

    dx = cv2.filter2D(src, -1, mx, borderType=cv2.BORDER_DEFAULT) + 128
    dy = cv2.filter2D(src, -1, my, borderType=cv2.BORDER_DEFAULT) + 128

    cv2.imshow("src", src)
    cv2.imshow("dx", dx)
    cv2.imshow("dy", dy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def sobel_edge():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    dx = cv2.Sobel(src, cv2.CV_32F, 1, 0)
    dy = cv2.Sobel(src, cv2.CV_32F, 0, 1)

    mag = cv2.magnitude(dx, dy)
    mag = cv2.convertScaleAbs(mag)
    edge = (mag > 150).astype(np.uint8) * 255

    cv2.imshow("src", src)
    cv2.imshow("mag", mag)
    cv2.imshow("edge", edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def canny_edge():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    dst1 = cv2.Canny(src, 50, 100)
    dst2 = cv2.Canny(src, 50, 150)

    cv2.imshow("src", src)
    cv2.imshow("dst1", dst1)
    cv2.imshow("dst2", dst2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    sobel_derivative()
    sobel_edge()
    canny_edge()