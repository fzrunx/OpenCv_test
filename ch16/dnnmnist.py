# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:25:41 2025

@author: user
"""

import cv2
import numpy as np

# 전역 변수
img = np.zeros((400, 400), dtype=np.uint8)
pt_prev = (-1, -1)

# 마우스 콜백
def on_mouse(event, x, y, flags, param):
    global pt_prev, img
    if event == cv2.EVENT_LBUTTONDOWN:
        pt_prev = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        pt_prev = (-1, -1)
    elif event == cv2.EVENT_MOUSEMOVE and (flags & cv2.EVENT_FLAG_LBUTTON):
        if pt_prev != (-1, -1):
            cv2.line(img, pt_prev, (x, y), 255, 40, cv2.LINE_AA)
            pt_prev = (x, y)
            cv2.imshow("img", img)

# DNN 모델 로드 (MNIST CNN .pb)
net = cv2.dnn.readNet("mnist_cnn.pb")
if net.empty():
    print("Network load failed!")
    exit()

cv2.namedWindow("img")
cv2.setMouseCallback("img", on_mouse)
cv2.imshow("img", img)

while True:
    key = cv2.waitKey(0)
    if key == 27:  # ESC 종료
        break
    elif key == ord(' '):  # 스페이스바: 예측
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0, size=(28,28))
        net.setInput(blob)
        prob = net.forward()
        digit = np.argmax(prob)
        confidence = prob[0][digit]
        print(f"{digit} ({confidence*100:.2f}%)")
        img[:] = 0
        cv2.imshow("img", img)

cv2.destroyAllWindows()