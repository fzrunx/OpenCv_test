# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:05:53 2025

@author: user
"""

import cv2

def detect_face():
    src = cv2.imread("kids.png")
    if src is None:
        print("Image load failed!")
        return

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    if face_cascade.empty():
        print("XML load failed!")
        return

    faces = face_cascade.detectMultiScale(src)
    for (x, y, w, h) in faces:
        cv2.rectangle(src, (x, y), (x+w, y+h), (255, 0, 255), 2)

    cv2.imshow("src", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detect_eyes():
    src = cv2.imread("kids.png")
    if src is None:
        print("Image load failed!")
        return

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")
    if face_cascade.empty() or eye_cascade.empty():
        print("XML load failed!")
        return

    faces = face_cascade.detectMultiScale(src)
    for (x, y, w, h) in faces:
        cv2.rectangle(src, (x, y), (x+w, y+h), (255, 0, 255), 2)
        faceROI = src[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(faceROI)
        for (ex, ey, ew, eh) in eyes:
            center = (ex + ew//2, ey + eh//2)
            cv2.circle(faceROI, center, ew//2, (255, 0, 0), 2)

    cv2.imshow("src", src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_face()
    detect_eyes()