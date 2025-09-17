# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 13:59:53 2025

@author: user
"""

import cv2
import numpy as np

def keypoint_matching():
    img1 = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Image load failed!")
        return

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)

    dst = cv2.drawMatches(img1, kp1, img2, kp2, matches, None)
    cv2.imshow("matches", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def good_matching():
    img1 = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Image load failed!")
        return

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]  # 상위 50개 선택

    dst = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("good matches", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def find_homography_example():
    img1 = cv2.imread("box.png", cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)
    if img1 is None or img2 is None:
        print("Image load failed!")
        return

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)[:50]

    dst = cv2.drawMatches(img1, kp1, img2, kp2, matches, None,
                          flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
    H, _ = cv2.findHomography(pts1, pts2, cv2.RANSAC)

    h, w = img1.shape
    corners1 = np.float32([[0,0], [w-1,0], [w-1,h-1], [0,h-1]]).reshape(-1,1,2)
    corners2 = cv2.perspectiveTransform(corners1, H)

    corners_dst = corners2 + np.array([ [ [w,0] ] ])  # 오른쪽으로 offset
    corners_dst = np.int32(corners_dst).reshape(-1,2)
    cv2.polylines(dst, [corners_dst], True, (0,255,0), 2, cv2.LINE_AA)

    cv2.imshow("homography", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    keypoint_matching()
    good_matching()
    find_homography_example()