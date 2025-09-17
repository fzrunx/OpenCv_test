# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:28:08 2025

@author: user
"""

import cv2
import numpy as np

def noise_gaussian():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    cv2.imshow("src", src)

    for stddev in range(10, 31, 10):
        noise = np.zeros(src.shape, dtype=np.int32)
        cv2.randn(noise, 0, stddev)

        dst = cv2.add(src, noise, dtype=cv2.CV_8U)

        desc = f"stddev = {stddev}"
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 255, 1, cv2.LINE_AA)
        cv2.imshow("dst", dst)
        cv2.waitKey()

    cv2.destroyAllWindows()


def filter_bilateral():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    noise = np.zeros(src.shape, dtype=np.int32)
    cv2.randn(noise, 0, 5)
    src = cv2.add(src, noise, dtype=cv2.CV_8U)

    dst1 = cv2.GaussianBlur(src, (0, 0), 5)
    dst2 = cv2.bilateralFilter(src, -1, 10, 5)

    cv2.imshow("src", src)
    cv2.imshow("dst1", dst1)
    cv2.imshow("dst2", dst2)

    cv2.waitKey()
    cv2.destroyAllWindows()


def filter_median():
    src = cv2.imread("lenna.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    num = int(src.size * 0.1)
    for i in range(num):
        x = np.random.randint(0, src.shape[1])
        y = np.random.randint(0, src.shape[0])
        src[y, x] = (i % 2) * 255

    dst1 = cv2.GaussianBlur(src, (0, 0), 1)
    dst2 = cv2.medianBlur(src, 3)

    cv2.imshow("src", src)
    cv2.imshow("dst1", dst1)
    cv2.imshow("dst2", dst2)

    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    noise_gaussian()
    filter_bilateral()
    filter_median()


if __name__ == "__main__":
    main()