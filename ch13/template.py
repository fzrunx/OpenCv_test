# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 12:12:56 2025

@author: user
"""

import cv2
import numpy as np

def template_matching():
    img = cv2.imread("circuit.bmp", cv2.IMREAD_COLOR)
    templ = cv2.imread("crystal.bmp", cv2.IMREAD_COLOR)

    if img is None or templ is None:
        print("Image load failed!")
        return

    # 밝기 증가
    img = cv2.add(img, (50, 50, 50, 0))

    # 노이즈 추가
    noise = np.zeros_like(img, dtype=np.int32)
    cv2.randn(noise, 0, 10)
    img = cv2.add(img, noise, dtype=cv2.CV_8UC3)

    # 템플릿 매칭
    res = cv2.matchTemplate(img, templ, cv2.TM_CCOEFF_NORMED)
    res_norm = cv2.normalize(res, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    print("max_val:", max_val)

    # 매칭 위치 표시
    top_left = max_loc
    bottom_right = (top_left[0] + templ.shape[1], top_left[1] + templ.shape[0])
    cv2.rectangle(img, top_left, bottom_right, (0, 0, 255), 2)

    cv2.imshow("templ", templ)
    cv2.imshow("res_norm", res_norm)
    cv2.imshow("img", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    template_matching()