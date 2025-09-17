# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:32:18 2025

@author: user
"""

import cv2
import numpy as np

# 마우스로 그리기
drawing = False
pt_prev = None

def on_mouse(event, x, y, flags, param):
    global drawing, pt_prev
    img = param
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        pt_prev = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        pt_prev = None
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.line(img, pt_prev, (x, y), (255,), 40, cv2.LINE_AA)
        pt_prev = (x, y)
        cv2.imshow('img', img)

# HOG + SVM 학습
def train_hog_svm():
    digits = cv2.imread('digits.png', cv2.IMREAD_GRAYSCALE)
    if digits is None:
        print("Image load failed!")
        return None

    hog = cv2.HOGDescriptor(_winSize=(20,20),
                            _blockSize=(10,10),
                            _blockStride=(5,5),
                            _cellSize=(5,5),
                            _nbins=9)

    train_hog = []
    train_labels = []

    for j in range(50):
        for i in range(100):
            roi = digits[j*20:(j+1)*20, i*20:(i+1)*20]
            desc = hog.compute(roi).flatten()
            train_hog.append(desc)
            train_labels.append(j // 5)

    train_hog = np.array(train_hog, dtype=np.float32)
    train_labels = np.array(train_labels, dtype=np.int32)

    svm = cv2.ml.SVM_create()
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setKernel(cv2.ml.SVM_RBF)
    svm.setC(2.5)
    svm.setGamma(0.50625)
    svm.train(train_hog, cv2.ml.ROW_SAMPLE, train_labels)

    return svm, hog

# 메인
img = np.zeros((400,400), dtype=np.uint8)
cv2.imshow('img', img)
cv2.setMouseCallback('img', on_mouse, img)

svm, hog = train_hog_svm()
if svm is None:
    exit()

while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC
        break
    elif key == ord(' '):  # 스페이스 → 예측
        img_resize = cv2.resize(img, (20,20))
        desc = hog.compute(img_resize).reshape(1,-1).astype(np.float32)
        res = int(svm.predict(desc)[1].ravel()[0])
        print(res)
        img[:] = 0
        cv2.imshow('img', img)

cv2.destroyAllWindows()