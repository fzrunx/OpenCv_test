# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:07:13 2025

@author: user
"""

import cv2
import random
import sys  # sys.exit() 사용 위해 import

cap = cv2.VideoCapture("vtest.avi")
if not cap.isOpened():
    print("Video open failed!")
    sys.exit()  # exit() 대신 sys.exit() 사용

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

while True:
    ret, frame = cap.read()
    if not ret:
        break

    detected, _ = hog.detectMultiScale(frame)

    for (x, y, w, h) in detected:
        color = (random.randint(0,255), random.randint(0,255), random.randint(0,255))
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)

    cv2.imshow("frame", frame)
    if cv2.waitKey(10) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()