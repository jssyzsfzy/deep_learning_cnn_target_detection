import xml.dom.minidom as xmldom
import os
import torch, gc
import torch.nn as nn
import torch.utils.data as Data
import cv2
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import time

classes = ['auto_car', 'raspberry', 'car']
num_class = len(classes)
LR = 0.00001
EPOCH = 100  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 2
sample_rate = 0.8
transf = transforms.ToTensor()


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 640, 640)
            nn.Conv2d(
                in_channels=3,  # input height
                out_channels=64,  # n_filters
                kernel_size=5,  # filter size
                stride=1,  # filter movement/step
                padding=2,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),  # output shape (16, 640, 640)
            nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 320, 320)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 320, 320)
            nn.Conv2d(64, 64, 5, 1, 0),  # output shape (32, 316, 316)
            nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 158, 158)
        )
        # there
        self.conv3 = nn.Sequential(  # input shape (32, 158, 158)
            nn.Conv2d(64, 80, 5, 1, 1),  # output shape (48, 156, 156)
            nn.BatchNorm2d(80, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(2),  # output shape (48, 78, 78)
        )
        self.conv4 = nn.Sequential(  # input shape (48, 78, 78)
            nn.Conv2d(80, 96, 3, 1, 0),  # output shape (64, 76, 76)
            nn.BatchNorm2d(96, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(2),  # output shape (64, 38, 38)
        )
        self.conv5 = nn.Sequential(  # input shape (64, 38, 38)
            nn.Conv2d(96, 128, 3, 1, 0),  # output shape (96, 36, 36)
            nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(2),  # output shape (32, 18, 18)
        )
        self.conv6 = nn.Sequential(  # input shape (96, 18, 18)
            nn.Conv2d(128, 256, 3, 1, 0),  # output shape (128, 16, 16)
            nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.AvgPool2d(2),
        )
        self.hidden1 = torch.nn.Linear(256 * 8 * 8, 512)  # 隐藏层线性输出
        self.hidden2 = torch.nn.Linear(512, 256)  # 隐藏层线性输出
        self.out = nn.Linear(256, 5 + num_class)  # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = x.view(x.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = F.silu(self.hidden1(x))  # 激励函数(隐藏层的线性值)
        x = torch.sigmoid(self.hidden2(x))
        output = self.out(x)
        return output  # return x for visualization


def parse_xml(fn):
    data = np.zeros((1, 1 + 4 + num_class))
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    try:
        name = eles.getElementsByTagName("name")[0].firstChild.data
        width = int(eles.getElementsByTagName("width")[0].firstChild.data)
        height = int(eles.getElementsByTagName("height")[0].firstChild.data)
        xmin = int(eles.getElementsByTagName("xmin")[0].firstChild.data)
        xmax = int(eles.getElementsByTagName("xmax")[0].firstChild.data)
        ymin = int(eles.getElementsByTagName("ymin")[0].firstChild.data)
        ymax = int(eles.getElementsByTagName("ymax")[0].firstChild.data)
        x_mean = (xmin + xmax) / (width * 2)
        y_mean = (ymin + ymax) / (height * 2)
        x_len = (xmax - xmin) / width
        y_len = (ymax - ymin) / height
        data[0][0:5] = [1, x_mean, y_mean, x_len, y_len]
        data[0][5 + classes.index(name)] = 1
    except:
        pass
    # print(fn, data)
    return data


def img_test(dir, save_dir, pkl_name, thes):
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
        output = cnn(train_data.cuda())
        if output[0][0] >= thes:
            cv2.rectangle(image,
                          (int((output[0][1] - output[0][3] / 2) * 640), int((output[0][2] - output[0][4] / 2) * 640)),
                          (int((output[0][1] + output[0][3] / 2) * 640), int((output[0][2] + output[0][4] / 2) * 640)),
                          (0, 255, 0), 2)
            media = output[0][5:].cpu().data.numpy()
            ind = media.tolist().index(max(media))
            cv2.putText(image, classes[ind],
                        (int((output[0][1] - output[0][3] / 2) * 640), int((output[0][2] - output[0][4] / 2) * 640)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        else:
            pass
        imag = cv2.resize(image, (shape[1], shape[0]))
        cv2.imwrite(save_dir + '/' + name_data[i], imag)
        print(save_dir + '/' + name_data[i] + ' have done')


def train(dir, load):
    name_data = os.listdir(dir)
    length = len(name_data)
    print(length)
    arr = np.arange(length)
    np.random.shuffle(arr)
    data = np.zeros((length, 1 + 4 + num_class))
    img_data = np.zeros((length, 3, 640, 640))
    for i in range(0, length):
        # print(i, arr[i])
        img = cv2.imread(dir + '/' + name_data[arr[i]])
        image = cv2.resize(img, (640, 640))
        img_tensor = transf(image)
        img_data[i] = img_tensor
        data[i] = parse_xml('img1_xml/' + name_data[arr[i]].split('.')[0] + '.xml')
    divide_num = int(length * sample_rate)
    train_data = torch.tensor(img_data[0:divide_num], dtype=torch.float32)
    Y_train_tensor = torch.tensor(data[0:divide_num], dtype=torch.float64)
    test_x = torch.tensor(img_data[divide_num:length], dtype=torch.float32)
    test_y = torch.tensor(data[divide_num:length], dtype=torch.float64)
    print(Y_train_tensor.shape)
    print(test_y.shape)
    torch_dataset = Data.TensorDataset(train_data, Y_train_tensor)
    torch_testset = Data.TensorDataset(test_x, test_y)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据
        num_workers=2,  # 多线程来读数据
    )
    test_data = Data.DataLoader(
        dataset=torch_testset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据
        num_workers=2  # 多线程来读数据
    )
    cnn = CNN().cuda()
    print(cnn)
    if load:
        cnn.load_state_dict(torch.load('net_one_img_more_class_cnn_params.pkl'))
    else:
        pass
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    print("start epoch")
    min_loss = 3
    for epoch in range(EPOCH):
        sum_loss = 0
        val_loss = 0
        start = time.time()
        for step, (x, y) in enumerate(loader):
            b_x = x.cuda()  # Tensor on GPU
            b_y = y.cuda()  # Tensor on GPU
            output = cnn(b_x)
            loss = loss_func(10 * output.float(), 10 * b_y.float())
            sum_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # gc.collect()
        # torch.cuda.empty_cache()
        with torch.no_grad():
            for step1, (val_x, val_y) in enumerate(test_data):
                x_val = val_x.cuda()
                y_val = val_y.cuda()
                test_output = cnn(x_val)
                loss_val = loss_func(10 * test_output.float(), 10 * y_val.float())
                val_loss += loss_val
        print("epoch is :", epoch, end='/' + str(EPOCH) + '\n')
        print("output", output)
        print("b_y", b_y)
        print("test_output", test_output)
        print("y_val", y_val)
        print("sum_loss:", sum_loss/divide_num)
        print("val_loss:", val_loss/(length - divide_num))
        if val_loss/(length - divide_num) < min_loss:
            min_loss = val_loss/(length - divide_num)
            # torch.save(cnn, 'net_one_img_more_target_cnn_ras_car.pkl')  # save entire net
            torch.save(cnn.state_dict(), 'net_one_img_more_class_cnn_params.pkl')  # save only the parameters
        print("min_loss:", min_loss)
        end = time.time()
        epoch_time = end - start
        print("Running time: %s seconds" % epoch_time)
        print('-' * 20)
    # torch.save(cnn, 'net_one_img_more_class_cnn.pkl')  # save entire net
    # torch.save(cnn.state_dict(), 'net_one_img_more_class_cnn_params.pkl')  # save only the parameters


def camera(pkl_name, threes):
    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load(pkl_name))
    img_data = np.zeros((1, 3, 640, 640))
    cap = cv2.VideoCapture(0)
    while True:
        gra, img = cap.read()
        shape = img.shape
        image = cv2.resize(img, (640, 640))
        img_tensor = transf(image)
        img_data[0] = img_tensor
        train_data = torch.tensor(img_data, dtype=torch.float32)
        output = cnn(train_data.cuda())
        if output[0][0] >= threes:
            cv2.rectangle(image,
                          (int((output[0][1] - output[0][3] / 2) * 640), int((output[0][2] - output[0][4] / 2) * 640)),
                          (int((output[0][1] + output[0][3] / 2) * 640), int((output[0][2] + output[0][4] / 2) * 640)),
                          (0, 255, 0), 2)
            media = output[0][5:].cpu().data.numpy()
            ind = media.tolist().index(max(media))
            cv2.putText(image, classes[ind],
                        (int((output[0][1] - output[0][3] / 2) * 640), int((output[0][2] - output[0][4] / 2) * 640)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        else:
            pass
        imag = cv2.resize(image, (shape[1], shape[0]))
        cv2.imshow("img", imag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break




if __name__ == "__main__":
    pkl = 'net_one_img_more_class_cnn_params.pkl'
    # img_test('save_camera', 'img_out', pkl, thes=0.85)
    # train(dir='img_640_more_class', load=False)
    camera(pkl_name=pkl, threes=0.85)
