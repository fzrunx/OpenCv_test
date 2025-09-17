# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:40:48 2025

@author: user
"""

import cv2
import sys
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions

# 이미지 불러오기
img_path = sys.argv[1] if len(sys.argv) > 1 else "space_shuttle.jpg"
img = cv2.imread(img_path)
if img is None:
    print("Image load failed!")
    sys.exit()

# Keras InceptionV3 모델 로드 (ImageNet)
model = InceptionV3(weights='imagenet')

# OpenCV 형식 -> RGB 변환 + 사이즈 맞춤
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img_rgb, (299, 299))

# 전처리
x = np.expand_dims(img_resized, axis=0)
x = preprocess_input(x)

# 예측
preds = model.predict(x)
decoded = decode_predictions(preds, top=1)[0][0]  # 상위 1개

label = f"{decoded[1]} ({decoded[2]*100:.2f}%)"
cv2.putText(img, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()