# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:29:14 2025

@author: user
"""

import cv2
import numpy as np

def unsharp_mask():
    src = cv2.imread("rose.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    cv2.imshow("src", src)

    for sigma in range(1, 6):
        blurred = cv2.GaussianBlur(src, (0, 0), sigma)

        alpha = 1.0
        dst = cv2.addWeighted(src, 1 + alpha, blurred, -alpha, 0)

        desc = f"sigma: {sigma}"
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 255, 1, cv2.LINE_AA)

        cv2.imshow("dst", dst)
        cv2.waitKey()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    unsharp_mask()