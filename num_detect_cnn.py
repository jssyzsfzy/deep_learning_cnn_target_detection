import cv2
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
import os
import torch
import torch.nn as nn
ball_color = 'red'
color_dist = {'red': {'Lower': np.array([0, 250, 200]), 'Upper': np.array([0, 255, 210])}}

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (3, 128, 128)
            nn.Conv2d(
                in_channels=3,              # input height
                out_channels=16,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 128, 128)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (16, 64, 64)
        )
        self.conv2 = nn.Sequential(         # input shape (16, 64, 64)
            nn.Conv2d(16, 32, 3, 1, 1),     # output shape (32, 64, 64)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (32, 32, 32)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 32, 32)
            nn.Conv2d(32, 48, 5, 1, 2),  # output shape (48, 32, 32)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (48, 16, 16)
        )
        self.conv4 = nn.Sequential(  # input shape (48, 16, 16)
            nn.Conv2d(48, 64, 3, 1, 1),  # output shape (64, 16, 16)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 8, 8)
        )
        self.hidden = torch.nn.Linear(64 * 8 * 8, 128)  # 隐藏层线性输出
        self.out = nn.Linear(128, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)       # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        output = self.out(x)
        return output, x    # return x for visualization

image = cv2.imread("detect.jpg")
# num = 7   # 采集数据集
# epo = 60  # 采集数据集
# [0, 255, 210]
img_data = np.zeros((1, 3, 128, 128))

# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# r, b = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)
[X, Y, D] = image.shape
gs_frame = cv2.GaussianBlur(image, (5, 5), 0)  # 高斯模糊
hsv = cv2.cvtColor(gs_frame, cv2.COLOR_BGR2HSV)  # 转化成HSV图像
erode_hsv = cv2.erode(hsv, None, iterations=2)  # 腐蚀 粗的变细
inRange_hsv = cv2.inRange(erode_hsv, color_dist[ball_color]['Lower'], color_dist[ball_color]['Upper'])
cnts = cv2.findContours(inRange_hsv.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
cnn = CNN().cuda()
print(cnn)

cnn.load_state_dict(torch.load('net_cnn_params.pkl'))

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
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    img = image[y_min:y_max, x_min:x_max]
    img_s = cv2.resize(img, (128, 128))
    transf = transforms.ToTensor()
    img_tensor = transf(img_s)  # tensor数据格式是torch(C,H,W)
    img_data[0] = img_tensor
    img_data1 = torch.tensor(img_data, dtype=torch.float32)
    test_output, last_layer = cnn(img_data1.cuda())
    pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
    txt = '{:.0f}'.format(pred_y.item())
    cv2.putText(image, txt, (x_min, y_min), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
cv2.imshow('camera', image)

cv2.waitKey(0)

cv2.destroyAllWindows()



