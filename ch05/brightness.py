# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:03:18 2025

@author: user
"""

import cv2
import numpy as np

def brightness1():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    dst = src + 100  # numpy 연산

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def brightness2():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    dst = np.zeros_like(src)

    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            dst[j, i] = src[j, i] + 100  # overflow 발생 가능

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def brightness3():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    dst = np.zeros_like(src)

    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            dst[j, i] = np.clip(src[j, i] + 100, 0, 255)  # saturate_cast와 동일

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def brightness4():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    def on_brightness(pos):
        dst = cv2.add(src, pos)  # OpenCV add는 자동으로 saturate 처리
        cv2.imshow("dst", dst)

    cv2.namedWindow("dst")
    cv2.createTrackbar("Brightness", "dst", 0, 100, on_brightness)
    on_brightness(0)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    brightness1()
    brightness2()
    brightness3()
    brightness4()