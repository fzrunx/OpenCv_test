# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:10:20 2025

@author: user
"""

import cv2

def decode_qrcode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera open failed!")
        return

    detector = cv2.QRCodeDetector()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame load failed!")
            break

        data, points, _ = detector.detectAndDecode(frame)

        if data:
            points = points.astype(int).reshape(-1, 2)
            cv2.polylines(frame, [points], True, (0, 0, 255), 2)
            cv2.putText(frame, data, (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 255))

        cv2.imshow("frame", frame)
        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    decode_qrcode()