# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 15:11:24 2025

@author: user
"""

import cv2
import sys  # <-- 추가

# 모델과 구성 파일 경로
model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"
config = "deploy.prototxt"

# 카메라 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera open failed!")
    sys.exit()  # <-- exit() 대신

# DNN 네트워크 불러오기
net = cv2.dnn.readNet(model, config)
if net.empty():
    print("Net open failed!")
    sys.exit()  # <-- exit() 대신

while True:
    ret, frame = cap.read()
    if not ret:
        break

    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104, 177, 123))
    net.setInput(blob)
    res = net.forward()

    for i in range(res.shape[2]):
        confidence = res[0, 0, i, 2]
        if confidence < 0.5:
            continue

        x1 = int(res[0, 0, i, 3] * frame.shape[1])
        y1 = int(res[0, 0, i, 4] * frame.shape[0])
        x2 = int(res[0, 0, i, 5] * frame.shape[1])
        y2 = int(res[0, 0, i, 6] * frame.shape[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Face: {confidence:.3f}"
        cv2.putText(frame, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()