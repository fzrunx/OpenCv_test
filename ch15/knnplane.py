# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:20:39 2025

@author: user
"""

import cv2
import numpy as np

# 전역 변수
img = np.zeros((500, 500, 3), dtype=np.uint8)
train = np.empty((0, 2), dtype=np.float32)
label = np.empty((0, 1), dtype=np.int32)
knn = cv2.ml.KNearest_create()
k_value = 1

# K 값 변경 콜백
def on_k_changed(pos):
    global k_value
    k_value = max(pos, 1)
    train_and_display()

# 학습 데이터 추가
def add_point(pt, cls):
    global train, label
    new_sample = np.array([[pt[0], pt[1]]], dtype=np.float32)
    train = np.vstack([train, new_sample])
    new_label = np.array([[cls]], dtype=np.int32)
    label = np.vstack([label, new_label])

# 학습 및 시각화
def train_and_display():
    if train.shape[0] == 0:
        return  # 데이터가 없으면 실행하지 않음

    knn.train(train, cv2.ml.ROW_SAMPLE, label)
    disp = np.zeros_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            sample = np.array([[j, i]], dtype=np.float32)
            _, _, res, _ = knn.findNearest(sample, k_value)
            response = int(res[0][0])
            if response == 0:
                disp[i, j] = (128, 128, 255)  # 빨강 영역
            elif response == 1:
                disp[i, j] = (128, 255, 128)  # 초록 영역
            else:
                disp[i, j] = (255, 128, 128)  # 파랑 영역

    # 학습 데이터 표시
    for i in range(train.shape[0]):
        x, y = int(train[i, 0]), int(train[i, 1])
        l = int(label[i, 0])
        color = [(0,0,128), (0,128,0), (128,0,0)][l]
        cv2.circle(disp, (x, y), 5, color, -1, cv2.LINE_AA)

    cv2.imshow('knn', disp)

# 초기 학습 데이터 생성
NUM = 30
rn = np.random.randn(NUM, 2) * 50
for i in range(NUM):
    add_point((int(rn[i,0]+150), int(rn[i,1]+150)), 0)
rn = np.random.randn(NUM, 2) * 50
for i in range(NUM):
    add_point((int(rn[i,0]+350), int(rn[i,1]+150)), 1)
rn = np.random.randn(NUM, 2) * 70
for i in range(NUM):
    add_point((int(rn[i,0]+250), int(rn[i,1]+400)), 2)

# 윈도우와 트랙바 설정
cv2.namedWindow('knn')
cv2.createTrackbar('k', 'knn', k_value, 5, on_k_changed)
train_and_display()

cv2.waitKey(0)
cv2.destroyAllWindows()