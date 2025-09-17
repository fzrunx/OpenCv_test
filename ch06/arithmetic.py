# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:20:06 2025

@author: user
"""

import cv2

def main():
    src1 = cv2.imread("lenna256.bmp", cv2.IMREAD_GRAYSCALE)
    src2 = cv2.imread("square.bmp", cv2.IMREAD_GRAYSCALE)

    if src1 is None or src2 is None:
        print("Image load failed!")
        return -1

    cv2.imshow("src1", src1)
    cv2.imshow("src2", src2)

    dst1 = cv2.add(src1, src2)
    dst2 = cv2.addWeighted(src1, 0.5, src2, 0.5, 0)
    dst3 = cv2.subtract(src1, src2)
    dst4 = cv2.absdiff(src1, src2)

    cv2.imshow("dst1", dst1)
    cv2.imshow("dst2", dst2)
    cv2.imshow("dst3", dst3)
    cv2.imshow("dst4", dst4)

    cv2.waitKey()
    return 0

if __name__ == "__main__":
    main()