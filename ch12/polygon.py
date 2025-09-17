# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:02:49 2025

@author: user
"""

import cv2
import numpy as np

def set_label(img, pts, label):
    x, y, w, h = cv2.boundingRect(pts)
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 1)
    cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))

img = cv2.imread("polygon.bmp")
if img is None:
    print("Image load failed!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, bin_img = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for pts in contours:
    if cv2.contourArea(pts) < 400:
        continue

    approx = cv2.approxPolyDP(pts, 0.02 * cv2.arcLength(pts, True), True)
    vtc = len(approx)

    if vtc == 3:
        set_label(img, pts, "TRI")
    elif vtc == 4:
        set_label(img, pts, "RECT")
    else:
        length = cv2.arcLength(pts, True)
        area = cv2.contourArea(pts)
        ratio = 4.0 * np.pi * area / (length * length)
        if ratio > 0.85:
            set_label(img, pts, "CIR")

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()