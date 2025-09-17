# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:33:08 2025

@author: user
"""

import cv2
import numpy as np

def affine_transform():
    src = cv2.imread("tekapo.bmp")
    if src is None:
        print("Image load failed!")
        return

    src_pts = np.float32([[0,0], [src.shape[1]-1,0], [src.shape[1]-1, src.shape[0]-1]])
    dst_pts = np.float32([[50,50], [src.shape[1]-100,100], [src.shape[1]-50, src.shape[0]-50]])

    M = cv2.getAffineTransform(src_pts, dst_pts)
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affine_translation():
    src = cv2.imread("tekapo.bmp")
    if src is None:
        print("Image load failed!")
        return

    M = np.float32([[1, 0, 150], [0, 1, 100]])
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affine_shear():
    src = cv2.imread("tekapo.bmp")
    if src is None:
        print("Image load failed!")
        return

    mx = 0.3
    M = np.float32([[1, mx, 0], [0, 1, 0]])
    width = int(src.shape[1] + src.shape[0]*mx)
    dst = cv2.warpAffine(src, M, (width, src.shape[0]))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affine_scale():
    src = cv2.imread("rose.bmp")
    if src is None:
        print("Image load failed!")
        return

    dst1 = cv2.resize(src, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    dst2 = cv2.resize(src, (1920, 1280))
    dst3 = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_CUBIC)
    dst4 = cv2.resize(src, (1920, 1280), interpolation=cv2.INTER_LANCZOS4)

    cv2.imshow("src", src)
    cv2.imshow("dst1", dst1[500:900, 400:800])
    cv2.imshow("dst2", dst2[500:900, 400:800])
    cv2.imshow("dst3", dst3[500:900, 400:800])
    cv2.imshow("dst4", dst4[500:900, 400:800])
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affine_rotation():
    src = cv2.imread("tekapo.bmp")
    if src is None:
        print("Image load failed!")
        return

    center = (src.shape[1]//2, src.shape[0]//2)
    M = cv2.getRotationMatrix2D(center, 20, 1)
    dst = cv2.warpAffine(src, M, (src.shape[1], src.shape[0]))

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def affine_flip():
    src = cv2.imread("eastsea.bmp")
    if src is None:
        print("Image load failed!")
        return

    cv2.imshow("src", src)
    flip_codes = [1, 0, -1]
    for code in flip_codes:
        dst = cv2.flip(src, code)
        cv2.putText(dst, f"flipCode: {code}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 1, cv2.LINE_AA)
        cv2.imshow("dst", dst)
        cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    affine_transform()
    affine_translation()
    affine_shear()
    affine_scale()
    affine_rotation()
    affine_flip()