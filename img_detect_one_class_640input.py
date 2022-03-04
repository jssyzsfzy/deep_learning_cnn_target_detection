import xml.dom.minidom as xmldom
import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import cv2
import numpy as np
import torchvision.transforms as transforms

classes = ['auto_car']
LR = 0.0005
EPOCH = 200  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 4
transf = transforms.ToTensor()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 640, 640)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=16,  # n_filters
                kernel_size=3,  # filter size
                stride=1,  # filter movement/step
                padding=1,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 640, 640)
            nn.ReLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 320, 320)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 320, 320)
            nn.Conv2d(16, 32, 3, 1, 1),  # output shape (32, 320, 320)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 160, 160)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 160, 160)
            nn.Conv2d(32, 48, 1, 1, 0),  # output shape (48, 160, 160)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (48, 80, 80)
        )
        self.conv4 = nn.Sequential(  # input shape (48, 80, 80)
            nn.Conv2d(48, 64, 5, 3, 0),  # output shape (64, 26, 26)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 13, 13)
        )
        self.conv5 = nn.Sequential(  # input shape (64, 13, 13)
            nn.Conv2d(64, 32, 3, 2, 1),  # output shape (32, 7, 7)
            nn.ReLU(),  # activation
        )
        self.conv6 = nn.Sequential(  # input shape (32, 7, 7)
            nn.Conv2d(32, 16, 3, 2, 0),  # output shape (32, 3, 3)
            nn.ReLU(),  # activation
        )
        self.out = nn.Sequential(  # input shape (16, 3, 3)
            nn.Conv2d(16, 6, 2, 1, 0),  # output shape (6, 2, 2)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (6, 1, 1)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        output = self.out(x)
        return output  # return x for visualization


def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    xmin = int(eles.getElementsByTagName("xmin")[0].firstChild.data)
    xmax = int(eles.getElementsByTagName("xmax")[0].firstChild.data)
    ymin = int(eles.getElementsByTagName("ymin")[0].firstChild.data)
    ymax = int(eles.getElementsByTagName("ymax")[0].firstChild.data)
    width = int(eles.getElementsByTagName("width")[0].firstChild.data)
    height = int(eles.getElementsByTagName("height")[0].firstChild.data)
    x_mean = (xmin + xmax) / (width * 2)
    y_mean = (ymin + ymax) / (height * 2)
    x_len = (xmax - xmin) / width
    y_len = (ymax - ymin) / height
    return x_mean, y_mean, x_len, y_len


def img_test(dir, save_dir, pkl_name):
    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load(pkl_name))
    name_data = os.listdir(dir)
    length = len(name_data)
    img_data = np.zeros((1, 3, 640, 640))
    for i in range(0, length):
        img = cv2.imread(dir + '/' + name_data[i])
        shape = img.shape
        image = cv2.resize(img, (640, 640))
        img_tensor = transf(image)
        img_data[0] = img_tensor
        train_data = torch.tensor(img_data, dtype=torch.float32)
        output = cnn(train_data.cuda()).view(-1, 6)
        rec = [int((output[0][1] - output[0][3] / 2) * 640), int((output[0][2] - output[0][4] / 2) * 640),
               int((output[0][1] + output[0][3] / 2) * 640), int((output[0][2] + output[0][4] / 2) * 640)]
        # print(np.multiply(np.array(rec), np.array([shape[1], shape[1], shape[0], shape[0]])))
        cv2.rectangle(image, (int((output[0][1] - output[0][3] / 2) * 640), int((output[0][2] - output[0][4] / 2) * 640)),
                      (int((output[0][1] + output[0][3] / 2) * 640), int((output[0][2] + output[0][4] / 2) * 640)), (0, 255, 0), 2)
        imag = cv2.resize(image, (shape[1], shape[0]))
        cv2.imwrite(save_dir + '/' + name_data[i], imag)
        print(save_dir + '/' + name_data[i] + ' have done')


def webcam(pkl_name):
    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load(pkl_name))
    img_data = np.zeros((1, 3, 640, 640))
    camera = cv2.VideoCapture(0)
    while True:
        gra, img = camera.read()
        image = cv2.resize(img, (640, 640))
        img_tensor = transf(image)
        img_data[0] = img_tensor
        train_data = torch.tensor(img_data, dtype=torch.float32)
        output = cnn(train_data.cuda()).view(-1, 6)
        cv2.rectangle(image,
                      (int((output[0][1] - output[0][3] / 2) * 640), int((output[0][2] - output[0][4] / 2) * 640)),
                      (int((output[0][1] + output[0][3] / 2) * 640), int((output[0][2] + output[0][4] / 2) * 640)),
                      (0, 255, 0), 2)
        cv2.imshow('camera', image)
        cv2.waitKey(1)


def train():
    name_data = os.listdir('img1')
    length = len(name_data)
    data = np.ones((length, 6))
    img_data = np.zeros((length, 3, 640, 640))
    for i in range(0, length):
        img = cv2.imread('img1/' + name_data[i])
        image = cv2.resize(img, (640, 640))
        img_tensor = transf(image)
        img_data[i] = img_tensor
        data[i][1:5] = parse_xml('img1_xml/' + name_data[i].split('.')[0] + '.xml')
    train_data = torch.tensor(img_data, dtype=torch.float32)
    Y_train_tensor = torch.tensor(data, dtype=torch.float64)
    torch_dataset = Data.TensorDataset(train_data, Y_train_tensor)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    cnn = CNN().cuda()
    print(cnn)
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    print("start epoch")
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader):
            b_x = x.cuda()  # Tensor on GPU
            b_y = y.cuda()  # Tensor on GPU
            output = cnn(b_x).view(-1, 6)
            loss = 100 * loss_func(output.float(), b_y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % 40 == 0:
                print("epoch is :", epoch)
                print("output", output)
                print("b_y", b_y)
                print("loss", loss)
                print('-' * 20)
    torch.save(cnn, 'net_img_cnn.pkl')  # save entire net
    torch.save(cnn.state_dict(), 'net_img_cnn_params.pkl')  # save only the parameters


if __name__ == "__main__":
    pkl = 'net_img_cnn_params.pkl'
    img_test('img1', 'img_out', pkl)
    # webcam(pkl)

