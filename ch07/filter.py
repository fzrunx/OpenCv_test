# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:26:49 2025

@author: user
"""

import cv2
import numpy as np

def filter_embossing():
    src = cv2.imread("rose.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    data = np.array([[-1, -1, 0],
                     [-1,  0, 1],
                     [ 0,  1, 1]], dtype=np.float32)
    emboss = data

    dst = cv2.filter2D(src, -1, emboss, anchor=(-1, -1), delta=128)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    filter_embossing()


if __name__ == "__main__":
    main()