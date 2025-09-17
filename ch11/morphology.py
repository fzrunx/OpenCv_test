# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:51:36 2025

@author: user
"""

import cv2
import numpy as np

def erode_dilate():
    src = cv2.imread("milkdrop.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    _, bin_img = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dst_erode = cv2.erode(bin_img, None)
    dst_dilate = cv2.dilate(bin_img, None)

    cv2.imshow("src", src)
    cv2.imshow("bin", bin_img)
    cv2.imshow("erode", dst_erode)
    cv2.imshow("dilate", dst_dilate)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def open_close():
    src = cv2.imread("milkdrop.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    _, bin_img = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    dst_open = cv2.morphologyEx(bin_img, cv2.MORPH_OPEN, None)
    dst_close = cv2.morphologyEx(bin_img, cv2.MORPH_CLOSE, None)

    cv2.imshow("src", src)
    cv2.imshow("bin", bin_img)
    cv2.imshow("opening", dst_open)
    cv2.imshow("closing", dst_close)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    erode_dilate()
    open_close()