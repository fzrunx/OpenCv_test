# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:02:08 2025

@author: user
"""

import cv2
import numpy as np

def labeling_basic():
    data = np.array([
        0, 0, 1, 1, 0, 0, 0, 0,
        1, 1, 1, 1, 0, 0, 1, 0,
        1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 1, 1, 0,
        0, 0, 0, 1, 1, 1, 1, 0,
        0, 0, 0, 1, 0, 0, 1, 0,
        0, 0, 1, 1, 1, 1, 1, 0,
        0, 0, 0, 0, 0, 0, 0, 0,
    ], dtype=np.uint8).reshape(8,8) * 255

    num_labels, labels = cv2.connectedComponents(data)

    print("src:\n", data)
    print("labels:\n", labels)
    print("number of labels:", num_labels)

def labeling_stats():
    src = cv2.imread("keyboard.bmp", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    _, bin_img = cv2.threshold(src, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_img)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)

    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area < 20:
            continue
        cv2.rectangle(dst, (x, y), (x+w, y+h), (0, 255, 255), 1)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    labeling_basic()
    labeling_stats()