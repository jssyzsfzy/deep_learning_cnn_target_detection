import cv2
import random
import torch.nn.functional as F
import numpy as np
import os
import torch
ball_color = 'red'
color_dist = {'red': {'Lower': np.array([0, 250, 200]), 'Upper': np.array([0, 255, 210])}}

class Net(torch.nn.Module):     # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()     # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)       # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))      # 激励函数(隐藏层的线性值)
        x = self.out(x)                 # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x



def pretreatment(image):
    ima = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    im=np.array(ima)        #转化为二维数组
    print(im.shape)
    for i in range(im.shape[0]):#转化为二值矩阵
        for j in range(im.shape[1]):
            if im[i,j]==63 or im[i,j]==62 or im[i,j]==64:
                im[i,j]=1
                # print("1")
            else:
                im[i,j]=0
    return im

image = cv2.imread("detect3.jpg")
# num = 7   # 采集数据集
# epo = 60  # 采集数据集
# [0, 255, 210]

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# r, b = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
[X, Y, D] = image.shape
gs_frame = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊
hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
net4 = Net(n_feature=32 * 32, n_hidden=10000, n_output=10)  # 几个类别就几个 output
net4.load_state_dict(torch.load('net2_par.pkl'))

for i in cnts:
    rect = cv2.minAreaRect(i)
    box = cv2.boxPoints(rect)
    # cv2.drawContours(image, [np.int0(box)], -1, (0, 0, 255), 2)
    x = [box[0][0], box[1][0], box[2][0], box[3][0]]
    y = [box[0][1], box[1][1], box[2][1], box[3][1]]
    x_min = int(min(x)-30) if(int(min(x)-30>0)) else 0
    x_max = int(max(x)+30) if(int(max(x)+30)<Y) else Y
    y_min = int(min(y)-30) if(int(min(y)-30)>0) else 0
    y_max = int(max(y)+30) if(int(max(y)+30)<X) else X
    # fron.append([x_min, x_max, y_min, y_max])
    img = image[y_min:y_max, x_min:x_max]
    cv2.imshow('camera', img)
    img_s = cv2.resize(img, (32, 32))

    im = pretreatment(img_s)

    X_test = np.reshape(im, (1, 1024))
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    out = net4(X_test_tensor)
    prediction = torch.max(F.softmax(out, dim=1), 1)[1]
    pred_y = prediction.data.numpy().squeeze()
    pred_y = prediction.data.numpy().squeeze()
    print(pred_y)
    cv2.waitKey(0)


cv2.destroyAllWindows()



