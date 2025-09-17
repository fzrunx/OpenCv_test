# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:25:17 2025

@author: user
"""

import cv2

def blurring_mean():
    src = cv2.imread("rose.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    cv2.imshow("src", src)

    for ksize in range(3, 8, 2):
        dst = cv2.blur(src, (ksize, ksize))

        desc = f"Mean: {ksize}x{ksize}"
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 255, 1, cv2.LINE_AA)

        cv2.imshow("dst", dst)
        cv2.waitKey()

    cv2.destroyAllWindows()


def blurring_gaussian():
    src = cv2.imread("rose.bmp", cv2.IMREAD_GRAYSCALE)

    if src is None:
        print("Image load failed!")
        return

    cv2.imshow("src", src)

    for sigma in range(1, 6):
        dst = cv2.GaussianBlur(src, (0, 0), sigma)

        desc = f"Gaussian: sigma = {sigma}"
        cv2.putText(dst, desc, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1.0, 255, 1, cv2.LINE_AA)

        cv2.imshow("dst", dst)
        cv2.waitKey()

    cv2.destroyAllWindows()


def main():
    blurring_mean()
    blurring_gaussian()


if __name__ == "__main__":
    main()