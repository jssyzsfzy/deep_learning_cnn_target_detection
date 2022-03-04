import torch
import numpy as np
import os
import cv2
import torch.nn as nn
import torch.utils.data as Data
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
file_name = 'img'
classes = ['auto_car', '1', '2', '0', '3', '4', '5', '6', '7', '8', '9']
EPOCH = 10               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 4
LR = 0.0001              # learning rate
transf = transforms.ToTensor()
def get_img2torch(files):
    name_data = os.listdir(files)
    length = len(name_data)
    arr = np.arange(length)
    np.random.shuffle(arr)
    name = []
    for i in range(0, length):
        name.append(name_data[arr[i]])
    label = np.ones((1, length))
    img_data = np.zeros((length, 3, 128, 128))
    for i in range(0, length):
        img = cv2.imread(files + '/' + name[i])
        img = cv2.resize(img, (128, 128))
        img_tensor = transf(img)
        img_data[i] = img_tensor
        label[0][i] = name[i].split('_')[0]
        # label[0][i] = classes.index([''.join(list(g)) for k, g in groupby(name[i], key=lambda x: x.isdigit())][0])
    return img_data, label, length

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


if __name__ == "__main__":
    data, y, l = get_img2torch(file_name)
    train_data = torch.tensor(data[0:l - 400], dtype=torch.float32)
    Y_train_tensor = torch.tensor(y[0][0:l - 400], dtype=torch.int64).view(1, -1).squeeze()
    test_x = torch.tensor(data[l - 400:l], dtype=torch.float32).cuda()
    test_y = torch.tensor(y[0][l - 400:l], dtype=torch.int64).view(1, -1).squeeze().cuda()
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
    cnn.load_state_dict(torch.load('net_cnn_params.pkl'))
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    print("start epoch")
    for epoch in range(EPOCH):
        for step, (x, y) in enumerate(loader):  # gives batch data, normalize x when iterate train_loader
            b_x = x.cuda()  # Tensor on GPU
            b_y = y.cuda()  # Tensor on GPU
            output = cnn(b_x)[0]
            loss = loss_func(output, b_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % 200 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
                print(pred_y)
                print(test_y)
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(),
                      '| test accuracy: %.2f' % accuracy)
    # torch.save(cnn, 'net_cnn.pkl')  # save entire net
    # torch.save(cnn.state_dict(), 'net_cnn_params.pkl')  # save only the parameters
    test_output, last_layer = cnn(test_x)
    pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
    accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / test_y.size(0)
    print(pred_y)
    print(test_y)
    print('test accuracy: %.2f' % accuracy)
