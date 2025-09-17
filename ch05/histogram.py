# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:14:35 2025

@author: user
"""

import cv2
import numpy as np

def calcGrayHist(img):
    assert img.dtype == np.uint8 and len(img.shape) == 2  # CV_8UC1 확인

    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist

def getGrayHistImage(hist):
    assert hist.dtype == np.float32
    assert hist.shape[0] == 256 and hist.shape[1] == 1

    histMax = hist.max()
    imgHist = np.full((100, 256), 255, dtype=np.uint8)

    for i in range(256):
        cv2.line(imgHist,
                 (i, 100),
                 (i, 100 - int(hist[i, 0] * 100 / histMax)),
                 0)

    return imgHist

def histogram_stretching():
    src = cv2.imread("hawkes.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    gmin, gmax, _, _ = cv2.minMaxLoc(src)
    dst = ((src - gmin) * 255.0 / (gmax - gmin)).astype(np.uint8)

    cv2.imshow("src", src)
    cv2.imshow("srcHist", getGrayHistImage(calcGrayHist(src)))

    cv2.imshow("dst", dst)
    cv2.imshow("dstHist", getGrayHistImage(calcGrayHist(dst)))

    cv2.waitKey()
    cv2.destroyAllWindows()

def histogram_equalization():
    src = cv2.imread("hawkes.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    dst = cv2.equalizeHist(src)

    cv2.imshow("src", src)
    cv2.imshow("srcHist", getGrayHistImage(calcGrayHist(src)))

    cv2.imshow("dst", dst)
    cv2.imshow("dstHist", getGrayHistImage(calcGrayHist(dst)))

    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    histogram_stretching()
    histogram_equalization()