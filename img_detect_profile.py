import xml.dom.minidom as xmldom
import torch
import torch.nn as nn
import torch.utils.data as Data
import cv2
import numpy as np
import torchvision.transforms as transforms
import time
import os
torch.cuda.set_device(0)

# print(torch.cuda.current_device())
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# import torch.nn.functional as F

classes = ['raspberry', 'car', 'bird']
num_class = len(classes)
LR = 0.0001
EPOCH = 300  # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 2
transf = transforms.ToTensor()
img_size = 640
xy_frame = 10
frame_size = 64
period = 64
skip = frame_size - period
sample_rate = 0.8


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # input shape (3, 640, 640)
            nn.Conv2d(
                in_channels=3,      # input height
                out_channels=32,    # n_filters
                kernel_size=3,      # filter size
                stride=1,           # filter movement/step
                padding=1,
                # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(kernel_size=2),  # choose max value in 2x2 area, output shape (16, 316, 316)
        )
        self.conv2 = nn.Sequential(  # input shape (16, 320, 320)
            nn.Conv2d(32, 48, 5, 1, 2),  # output shape (32, 320, 320)
            nn.BatchNorm2d(48, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.AvgPool2d(2),                # output shape (32, 160, 160)
        )
        self.conv3 = nn.Sequential(  # input shape (32, 320, 320)
            nn.Conv2d(48, 64, 5, 1, 2),  # output shape (64, 320, 320)
            nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.AvgPool2d(2),
        )
        self.conv4 = nn.Sequential(  # input shape (64, 320, 320)
            nn.Conv2d(64, 128, 3, 1, 1),  # output shape (128, 320, 320)
            nn.BatchNorm2d(128, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.AvgPool2d(2),
        )
        self.conv5 = nn.Sequential(  # input shape (128, 320, 320)
            nn.Conv2d(128, 160, 1, 1, 0),  # output shape (96, 5, 5)
            nn.BatchNorm2d(160, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
            nn.MaxPool2d(2),
        )
        self.conv6 = nn.Sequential(  # input shape (128, 5, 5)
            nn.Conv2d(160, 192, 5, 1, 0),  # output shape (96, 5, 5)
            nn.BatchNorm2d(192, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )
        self.conv7 = nn.Sequential(  # input shape (128, 5, 5)
            nn.Conv2d(192, 256, 3, 1, 0),  # output shape (96, 5, 5)
            nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )
        self.conv8 = nn.Sequential(  # input shape (128, 5, 5)
            nn.Conv2d(256, 400, 3, 1, 0),  # output shape (96, 5, 5)
            nn.BatchNorm2d(400, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )
        self.conv9 = nn.Sequential(  # input shape (128, 5, 5)
            nn.Conv2d(400, 512, 3, 1, 0),  # output shape (96, 5, 5)
            nn.BatchNorm2d(512, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )
        self.conv10 = nn.Sequential(  # input shape (128, 5, 5)
            nn.Conv2d(512, 560, 1, 1, 0),  # output shape (96, 5, 5)
            nn.BatchNorm2d(560, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )
        self.conv11 = nn.Sequential(  # input shape (128, 5, 5)
            nn.Conv2d(560, 256, 1, 1, 0),  # output shape (96, 5, 5)
            nn.BatchNorm2d(256, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )
        self.conv12 = nn.Sequential(  # input shape (128, 5, 5)
            nn.Conv2d(256, 64, 1, 1, 0),  # output shape (96, 5, 5)
            # nn.BatchNorm2d(64, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )
        self.out = nn.Sequential(  # input shape (24, 3, 3)
            nn.Conv2d(64, 5 + num_class, 1, 1, 0),  # output shape (7, 2, 2)
            # nn.BatchNorm2d(5 + num_class, eps=0.001, momentum=0.03, affine=True, track_running_stats=True),
            nn.SiLU(),  # activation
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.conv11(x)
        x = self.conv12(x)
        output = self.out(x)
        return 3 * output  # return x for visualization


def parse_xml(fn):
    xml_file = xmldom.parse(fn)
    eles = xml_file.documentElement
    data = np.zeros((1 + 4 + num_class, xy_frame, xy_frame))
    try:
        name = eles.getElementsByTagName("name")
        length = len(name)
        # print(fn, length, name)
        width = int(eles.getElementsByTagName("width")[0].firstChild.data)
        height = int(eles.getElementsByTagName("height")[0].firstChild.data)
        for i in range(0, length):
            xmin = int(eles.getElementsByTagName("xmin")[i].firstChild.data)
            xmax = int(eles.getElementsByTagName("xmax")[i].firstChild.data)
            x = (xmin + xmax) * (img_size/2) / width
            x_in = int((x - skip/2) / period)
            ymin = int(eles.getElementsByTagName("ymin")[i].firstChild.data)
            ymax = int(eles.getElementsByTagName("ymax")[i].firstChild.data)
            y = (ymin + ymax) * (img_size/2) / height
            y_in = int((y - skip/2) / period)
            name = eles.getElementsByTagName("name")[i].firstChild.data
            x_mean = (x - period * x_in) / frame_size
            y_mean = (y - period * y_in) / frame_size
            x_len = (xmax - xmin) * img_size / (frame_size * width)
            y_len = (ymax - ymin) * img_size / (frame_size * height)
            # print(fn, x_in, y_in, xmin, ymin, xmax, ymax, width, height, name, x_mean, y_mean, x_len, y_len)
            t = [1, x_mean, y_mean, x_len, y_len]
            print(fn, t)
            for j in range(0, 5):
                # if x_in == xy_frame:
                #     x_in = xy_frame - 1
                # if y_in == xy_frame:
                #     y_in = xy_frame - 1
                data[j][x_in][y_in] = t[j]
            data[5 + classes.index(name)][x_in][y_in] = 1
    except:
        pass
    return data


def img_test(dir, save_dir, pkl_name, threes):
    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load(pkl_name))
    name_data = os.listdir(dir)
    length = len(name_data)
    img_data = np.zeros((1, 3, img_size, img_size))
    for i in range(0, length):
        img = cv2.imread(dir + '/' + name_data[i])
        shape = img.shape
        image = cv2.resize(img, (img_size, img_size))
        img_tensor = transf(image)
        img_data[0] = img_tensor
        train_data = torch.tensor(img_data, dtype=torch.float32)
        output = cnn(train_data.cuda()).cpu().data.numpy()
        print(output)
        index = np.where(output[0][0] > threes)
        data = np.zeros((len(index[0]), 4 + 1))
        for j in range(0, len(index[0])):
            data[j][0] = output[0][1][index[0][j]][index[1][j]] * frame_size + period * index[0][j]
            data[j][1] = output[0][2][index[0][j]][index[1][j]] * frame_size + period * index[1][j]
            data[j][2] = output[0][3][index[0][j]][index[1][j]] * frame_size
            data[j][3] = output[0][4][index[0][j]][index[1][j]] * frame_size
            out = output[0, 5:, index[0][j], index[1][j]]
            data[j][4] = np.argwhere(out == np.max(out))
        # print(data)
        for j in range(0, len(data)):
            cv2.rectangle(image, (int(data[j][0] - data[j][2] / 2), int(data[j][1] - data[j][3] / 2)),
                          (int(data[j][0] + data[j][2] / 2), int(data[j][1] + data[j][3] / 2)), (0, 255, 0), 2)
            cv2.putText(image, classes[int(data[j][4])],
                        (int(data[j][0] - data[j][2] / 2), int(data[j][1] - data[j][3] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        imag = cv2.resize(image, (shape[1], shape[0]))
        cv2.imwrite(save_dir + '/' + name_data[i], imag)
        print(save_dir + '/' + name_data[i] + ' have done')
        print('-' * 20)


def webcam(pkl_name, threes):
    cnn = CNN().cuda()
    cnn.load_state_dict(torch.load(pkl_name))
    img_data = np.zeros((1, 3, img_size, img_size))
    cap = cv2.VideoCapture(1)
    while True:
        gra, img = cap.read()
        shape = img.shape
        image = cv2.resize(img, (img_size, img_size))
        img_tensor = transf(image)
        img_data[0] = img_tensor
        train_data = torch.tensor(img_data, dtype=torch.float32)
        output = cnn(train_data.cuda()).cpu().data.numpy()
        index = np.where(output[0][0] > threes)
        data = np.zeros((len(index[0]), 4 + 1))
        for j in range(0, len(index[0])):
            data[j][0] = output[0][1][index[0][j]][index[1][j]] * frame_size + period * index[0][j]
            data[j][1] = output[0][2][index[0][j]][index[1][j]] * frame_size + period * index[1][j]
            data[j][2] = output[0][3][index[0][j]][index[1][j]] * frame_size
            data[j][3] = output[0][4][index[0][j]][index[1][j]] * frame_size
            out = output[0, 5:, index[0][j], index[1][j]]
            data[j][4] = np.argwhere(out == np.max(out))
        # print(data)
        for j in range(0, len(data)):
            cv2.rectangle(image, (int(data[j][0] - data[j][2] / 2), int(data[j][1] - data[j][3] / 2)),
                          (int(data[j][0] + data[j][2] / 2), int(data[j][1] + data[j][3] / 2)), (0, 255, 0), 2)
            cv2.putText(image, classes[int(data[j][4])],
                        (int(data[j][0] - data[j][2] / 2), int(data[j][1] - data[j][3] / 2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
        imag = cv2.resize(image, (shape[1], shape[0]))
        cv2.imshow("img", imag)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


def train(load):
    f = open('datasets/ImageSets_more_target/train.txt', 'r')
    lines = f.readlines()
    length_train = len(lines)
    print(length_train)
    data = np.zeros((length_train, 1 + 4 + num_class, xy_frame, xy_frame))
    img_data = np.zeros((length_train, 3, img_size, img_size))
    num_read = 0
    for line in lines:
        # print(line.strip())
        img = cv2.imread(line.strip())
        image = cv2.resize(img, (img_size, img_size))
        img_tensor = transf(image)
        img_data[num_read] = img_tensor
        data[num_read] = parse_xml('img1_xml/' + line.strip().split('/')[1].split('.')[0] + '.xml')
        num_read += 1
    print(num_read)
    f.close()
    f = open('datasets/ImageSets_more_target/val.txt')
    lines_val = f.readlines()
    length_val = len(lines_val)
    print(length_val)
    data_val = np.zeros((length_val, 1 + 4 + num_class, xy_frame, xy_frame))
    img_val = np.zeros((length_val, 3, img_size, img_size))
    num_read = 0
    for line in lines_val:
        # print(line.strip())
        img = cv2.imread(line.strip())
        image = cv2.resize(img, (img_size, img_size))
        img_tensor = transf(image)
        img_val[num_read] = img_tensor
        data_val[num_read] = parse_xml('img1_xml/' + line.strip().split('/')[1].split('.')[0] + '.xml')
        num_read += 1
    print(num_read)
    f.close()
    train_data = torch.tensor(img_data, dtype=torch.float32)
    Y_train_tensor = torch.tensor(data, dtype=torch.float64)
    test_x = torch.tensor(img_val, dtype=torch.float32)
    test_y = torch.tensor(data_val, dtype=torch.float64)
    torch_dataset = Data.TensorDataset(train_data, Y_train_tensor)
    torch_testset = Data.TensorDataset(test_x, test_y)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据
        num_workers=0,  # 多线程来读数据
    )
    test_data = Data.DataLoader(
        dataset=torch_testset,  # torch TensorDataset format
        batch_size=1,  # mini batch size
        shuffle=True,  # 要不要打乱数据
        num_workers=0  # 多线程来读数据
    )
    cnn = CNN().cuda()
    print(cnn)
    if load:
        cnn.load_state_dict(torch.load('net_one_img_more_target_cnn_params_ras_car.pkl'))
    else:
        pass
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.MSELoss()  # the target label is not one-hotted
    min_loss = 10000
    print("start epoch")
    for epoch in range(EPOCH):
        sum_loss = 0
        val_loss = 0
        start = time.time()
        for step, (x, y) in enumerate(loader):
            b_x = x.cuda()  # Tensor on GPU
            b_y = y.cuda()  # Tensor on GPU
            # print(b_y.shape)
            # print(b_x.shape)
            output = cnn(b_x)
            loss = loss_func(100 * output.float(), 100 * b_y.float())
            sum_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # if step % 200 == 0:
        with torch.no_grad():
            for step1, (val_x, val_y) in enumerate(test_data):
                x_val = val_x.cuda()
                y_val = val_y.cuda()
                test_output = cnn(x_val)
                loss_val = loss_func(100 * test_output.float(), 100 * y_val.float())
                val_loss += loss_val
        if (epoch + 1) % 3 == 0:
            # print("output", output[0][0])
            # print("b_y", b_y[0][0])
            print("test_output", test_output[0][0])
            print("y_val", y_val[0][0])
        print("epoch is :", epoch + 1, end='/' + str(EPOCH) + '\n')
        print("sum_loss:", sum_loss, ' ', sum_loss / length_train)
        print("val_loss:", val_loss, ' ', val_loss / length_val)
        if val_loss / length_val < min_loss:
            min_loss = val_loss / length_val
            # torch.save(cnn, 'net_one_img_more_target_cnn_ras_car.pkl')  # save entire net
            torch.save(cnn.state_dict(), 'net_one_img_more_target_cnn_params_ras_car.pkl')  # save only the parameters
        end = time.time()
        epoch_time = end - start
        print("Running time: %s seconds" % (epoch_time))
        # print("还剩时间为：", )
        print('-' * 20)
    # torch.save(cnn, 'net_one_img_more_target_cnn.pkl')  # save entire net
    # torch.save(cnn.state_dict(), 'net_one_img_more_target_cnn_params.pkl')  # save only the parameters


if __name__ == "__main__":
    pkl = 'net_one_img_more_target_cnn_params_ras_car.pkl'
    # img_test('save_camera', 'img_out', pkl, threes=0.85)
    train(load=False)
    # webcam(pkl, threes=0.8)
