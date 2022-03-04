import cv2
import random
import torch.nn.functional as F
import numpy as np
import os
import torch

ball_color = 'green'
color_dist = {'red': {'Lower': np.array([0, 250, 200]), 'Upper': np.array([0, 255, 210])},
              'orange': {'Lower': np.array([15, 210, 240]), 'Upper': np.array([20, 225, 255])},
              'blue': {'Lower': np.array([100, 250, 250]), 'Upper': np.array([110, 255, 255])},
              'pongue': {'Lower': np.array([140, 190, 135]), 'Upper': np.array([150, 200, 145])},
              'black': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([2, 2, 2])},
              'green': {'Lower': np.array([55, 250, 250]), 'Upper': np.array([65, 255, 255])},
              'white': {'Lower': np.array([0, 0, 250]), 'Upper': np.array([0, 0, 255])},
              }
image = cv2.imread("img_origin/IMG_2056(20220216-160536).PNG")
num = 9  # 采集数据集
epo = 972  # 采集数据集
# [0, 255, 210]

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# r, b = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
[X, Y, D] = image.shape
gs_frame = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊
hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

for i in cnts:
    rect = cv2.minAreaRect(i)
    box = cv2.boxPoints(rect)
    # cv2.drawContours(image, [np.int0(box)], -1, (0, 0, 255), 2)
    x = [box[0][0], box[1][0], box[2][0], box[3][0]]
    y = [box[0][1], box[1][1], box[2][1], box[3][1]]
    x_min = int(min(x) - 30) if (int(min(x) - 30 > 0)) else 0
    x_max = int(max(x) + 30) if (int(max(x) + 30) < Y) else Y
    y_min = int(min(y) - 30) if (int(min(y) - 30) > 0) else 0
    y_max = int(max(y) + 30) if (int(max(y) + 30) < X) else X
    # fron.append([x_min, x_max, y_min, y_max])
    img = image[y_min:y_max, x_min:x_max]
    # cv2.imshow('camera', img)
    img_s = cv2.resize(img, (128, 128))
    # 采集数据集
    cv2.imwrite('img/' + str(num) + '_' + str(epo) + '.jpg', img_s)
    epo += 1  # 采集数据集
    # cv2.waitKey(0)
#


# cv2.imshow('camera', c)
# img = pretreatment(image)
#
#
# print(fron)
# cv2.imshow("image", b)

print(epo)
cv2.waitKey(0)

cv2.destroyAllWindows()
