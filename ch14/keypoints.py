# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:17:51 2025

@author: user
"""

import cv2

def detect_keypoints():
    src = cv2.imread("box_in_scene.png", cv2.IMREAD_GRAYSCALE)
    if src is None:
        print("Image load failed!")
        return

    orb = cv2.ORB_create()
    keypoints, desc = orb.detectAndCompute(src, None)

    print("keypoints.size():", len(keypoints))
    if desc is not None:
        print("desc.shape:", desc.shape)

    dst = cv2.drawKeypoints(src, keypoints, None, color=(0,0,255),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow("src", src)
    cv2.imshow("dst", dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    detect_keypoints()