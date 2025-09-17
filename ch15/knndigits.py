# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:12:24 2025

@author: user
"""

import cv2
import numpy as np

pt_prev = None

# -------------------
# 마우스 콜백
def on_mouse(event, x, y, flags, param):
    global pt_prev
    img = param

    if event == cv2.EVENT_LBUTTONDOWN:
        pt_prev = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        pt_prev = None
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        if pt_prev is not None:
            cv2.line(img, pt_prev, (x, y), 255, 40, cv2.LINE_AA)
            pt_prev = (x, y)
            cv2.imshow("img", img)

# -------------------
# KNN 학습
digits = cv2.imread("digits.png", cv2.IMREAD_GRAYSCALE)
if digits is None:
    print("Image load failed!")
    exit()

train_images = []
train_labels = []

for j in range(50):
    for i in range(100):
        roi = digits[j*20:j*20+20, i*20:i*20+20]
        roi_flatten = roi.reshape(-1).astype(np.float32)
        train_images.append(roi_flatten)
        train_labels.append(j // 5)

train_images = np.array(train_images, dtype=np.float32)
train_labels = np.array(train_labels, dtype=np.float32)

knn = cv2.ml.KNearest_create()
knn.train(train_images, cv2.ml.ROW_SAMPLE, train_labels)

# -------------------
# 그리기
img = np.zeros((400, 400), dtype=np.uint8)
cv2.imshow("img", img)
cv2.setMouseCallback("img", on_mouse, img)

while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == 32:  # Space
        roi_resize = cv2.resize(img, (20, 20), interpolation=cv2.INTER_AREA)
        roi_flatten = roi_resize.reshape(1, -1).astype(np.float32)
        _, result, _, _ = knn.findNearest(roi_flatten, 3)
        print(int(result[0,0]))
        img[:] = 0
        cv2.imshow("img", img)

cv2.destroyAllWindows()