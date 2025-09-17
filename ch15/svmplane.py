# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:36:03 2025

@author: user
"""

import cv2
import numpy as np

# 학습 데이터
train = np.array([
    [150, 200], [200, 250], [100, 250], [150, 300],
    [350, 100], [400, 200], [400, 300], [350, 400]
], dtype=np.float32)
label = np.array([[0], [0], [0], [0], [1], [1], [1], [1]], dtype=np.int32)

# SVM 학습
svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_RBF)
svm.trainAuto(train, cv2.ml.ROW_SAMPLE, label)

# 결과 이미지
img = np.zeros((500, 500, 3), dtype=np.uint8)

for y in range(img.shape[0]):
    for x in range(img.shape[1]):
        sample = np.array([[x, y]], dtype=np.float32)
        res = svm.predict(sample)[1]
        if int(res[0, 0]) == 0:
            img[y, x] = (128, 128, 255)  # 빨강 영역
        else:
            img[y, x] = (128, 255, 128)  # 초록 영역

# 학습 데이터 표시
for i in range(train.shape[0]):
    x, y = int(train[i, 0]), int(train[i, 1])
    l = int(label[i, 0])
    color = (0, 0, 128) if l == 0 else (0, 128, 0)
    cv2.circle(img, (x, y), 5, color, -1, cv2.LINE_AA)

cv2.imshow("svm", img)
cv2.waitKey(0)
cv2.destroyAllWindows()