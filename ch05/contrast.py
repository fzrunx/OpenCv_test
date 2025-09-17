# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:11:54 2025

@author: user
"""

import cv2
import numpy as np

def contrast1():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    s = 2.0
    dst = src.astype(np.float32) * s   # 곱하면 float 되므로
    dst = np.clip(dst, 0, 255).astype(np.uint8)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


def contrast2():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    alpha = 1.0
    dst = src.astype(np.float32) + (src.astype(np.float32) - 128) * alpha
    dst = np.clip(dst, 0, 255).astype(np.uint8)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    contrast1()
    contrast2()