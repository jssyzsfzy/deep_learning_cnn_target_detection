"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
torch: 0.4
torchvision
matplotlib
"""
# library
# standard library
import os
import numpy as np
# third-party library
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# torch.manual_seed(1)    # reproducible
name_list = 'data1'
# Hyper Parameters
EPOCH = 1               # train the training data n times, to save time, we just train 1 epoch
BATCH_SIZE = 10
LR = 0.001              # learning rate

def get_data(filename):
    f = open(filename)
    returnVect = np.zeros((32, 32))
    for i in range(32):
        line_data = f.readline()
        for j in range(32):
            returnVect[i, j] = int(line_data[j])
    return returnVect

def get_img_data():
    name = os.listdir(name_list)
    length = len(name)
    arr = np.arange(length)
    np.random.shuffle(arr)
    name_get = []
    for i in range(0, length-1):
        name_get.append(name[arr[i]])
    data = np.ones((length, 1, 32, 32))
    label = np.zeros((length))
    for i in range(0, length-1):
        # print(name_list + '/' + name_get[i])
        data[i][0] = get_data(name_list + '/' + name_get[i])
        label[i] = name_get[i].split('_')[0]
    y = np.array(label).reshape(length, 1)
    print("start")
    return data, y, length
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(         # input shape (1, 32, 32)
            nn.Conv2d(
                in_channels=1,              # input height
                out_channels=32,            # n_filters
                kernel_size=3,              # filter size
                stride=1,                   # filter movement/step
                padding=1,                  # if want same width and length of this image after Conv2d, padding=(kernel_size-1)/2 if stride=1
            ),                              # output shape (16, 28, 28)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(kernel_size=2),    # choose max value in 2x2 area, output shape (32, 16, 16)
        )
        self.conv2 = nn.Sequential(         # input shape (32, 16, 16)
            nn.Conv2d(32, 48, 5, 1, 2),     # output shape (48, 16, 16)
            nn.ReLU(),                      # activation
            nn.MaxPool2d(2),                # output shape (48, 8, 8)
        )
        self.out = nn.Linear(48 * 8 * 8, 10)   # fully connected layer, output 10 classes

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        output = self.out(x)
        return output, x    # return x for visualization


if __name__ == "__main__":
    data, y, length = get_img_data()
    # 先转换成 torch 能识别的 Dataset
    train_data = torch.tensor(data[0:length - 50], dtype=torch.float32)
    Y_train_tensor = torch.tensor(y[0:length - 50], dtype=torch.int64)
    test_x = torch.tensor(data[length - 50:length], dtype=torch.float32).cuda()
    test_y = torch.tensor(y[length - 50:length], dtype=torch.int64).cuda()
    torch_dataset = Data.TensorDataset(train_data, Y_train_tensor)
    # 把 dataset 放入 DataLoader
    loader = Data.DataLoader(
        dataset=torch_dataset,  # torch TensorDataset format
        batch_size=BATCH_SIZE,  # mini batch size
        shuffle=True,  # 要不要打乱数据 (打乱比较好)
        num_workers=2,  # 多线程来读数据
    )
    cnn = CNN().cuda()
    # print(cnn)  # net architecture
    """
    CNN(
      (conv1): Sequential(
        (0): Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (conv2): Sequential(
        (0): Conv2d(16, 48, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2))
        (1): ReLU()
        (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
      )
      (out): Linear(in_features=3072, out_features=10, bias=True)
    )
    """
    optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters
    loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted
    print("start epoch")
    for epoch in range(100):
        for step, (x, y) in enumerate(loader):   # gives batch data, normalize x when iterate train_loader
            b_x = x.cuda()
            b_y = y.cuda()
            target = b_y.view(1, -1)  # 转换为1维
            target = target.squeeze().cuda()
            output = cnn(b_x)[0]  # cnn output
            loss = loss_func(output, target)  # cross entropy loss
            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients
            if step % 200 == 0:
                test_output, last_layer = cnn(test_x)
                pred_y = torch.max(test_output, 1)[1].cuda().data  # move the computation in GPU
                accuracy = torch.sum(pred_y == test_y).type(torch.FloatTensor) / (test_y.size(0)*50)
                print('Epoch: ', epoch, '| train loss: %.4f' % loss.cpu().data.numpy(), '| test accuracy: %.2f' % accuracy)
    torch.save(cnn, 'net_cnn.pkl')  # save entire net
    torch.save(cnn.state_dict(), 'net_cnn_params.pkl')  # save only the parameters



