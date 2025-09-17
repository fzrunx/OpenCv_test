# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 11:40:34 2025

@author: user
"""

import cv2
import numpy as np

def hough_lines():
    src = cv2.imread("building.jpg", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    edge = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLines(edge, 1, np.pi/180, 250)

    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for rho, theta in lines[:,0]:
            a, b = np.cos(theta), np.sin(theta)
            x0, y0 = a*rho, b*rho
            alpha = 1000
            pt1 = (int(x0 - alpha*b), int(y0 + alpha*a))
            pt2 = (int(x0 + alpha*b), int(y0 - alpha*a))
            cv2.line(dst, pt1, pt2, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough_line_segments():
    src = cv2.imread("building.jpg", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    edge = cv2.Canny(src, 50, 150)
    lines = cv2.HoughLinesP(edge, 1, np.pi/180, 160, minLineLength=50, maxLineGap=5)

    dst = cv2.cvtColor(edge, cv2.COLOR_GRAY2BGR)
    if lines is not None:
        for x1, y1, x2, y2 in lines[:,0]:
            cv2.line(dst, (x1,y1), (x2,y2), (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def hough_circles():
    src = cv2.imread("coins.png", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    blurred = cv2.blur(src, (3,3))
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 50, param1=150, param2=30)

    dst = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for x, y, r in circles[0,:]:
            cv2.circle(dst, (x,y), r, (0,0,255), 2, cv2.LINE_AA)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    hough_lines()
    hough_line_segments()
    hough_circles()