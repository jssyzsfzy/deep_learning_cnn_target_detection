# 1支持向量机
import random
import torch.nn.functional as F
import numpy as np
import os
import torch

def img2vector(filename):
    # 创建向量
    returnVect = np.zeros((1, 1024))
    # 打开数据文件,读取每行内容
    fr = open(filename)
    for i in range(32):
        # 读取每一行
        lineStr = fr.readline()
        # 将每行前32字符转成int,存入向量
        for j in range(32):
            returnVect[0, 32 * i + j] = int(lineStr[j])
    return returnVect


def trainData(trainPath):
    trainfile = os.listdir(trainPath)  # 获取训练集文件下的所有文件名
    Y = np.zeros((len(trainfile), 1))
    # 先建立一个行数为训练样本数。列数为1024的0数组矩阵，1024为图片像素总和，即32*32
    X = np.zeros((len(trainfile), 1024))
    size = [[], [], [], [], [], [], [], [], [], []]
    # 取文件名的第一个数字为标签名
    for i in range(0, len(trainfile)):
        thislabel = trainfile[i].split(".")[0].split("_")[0]
        if len(thislabel) != 0:
            size[int(thislabel)].append(i)
            Y[i][0] = int(thislabel)  # 保存标签
        X[i, :] = img2vector(trainPath + "/" + trainfile[i])  # 将训练数据写入0矩阵
    return X, Y, size



X, Y, size = trainData('data1')
ran = []
for i in range(len(size)):
    ran.append(random.sample(size[i], len(size[i])))

fen = [[], [], [], [], []]
for i in range(len(ran)):
    length = len(ran[i])
    n = 5
    step = int(length / n) + 1
    num = 0
    for j in range(0, length, step):
        fen[num].extend(ran[i][j: j + step])
        num += 1

X_part = []
Y_part = []
print(len(Y))
for i in range(0, 5):
    temp = []
    temp1 = []
    for j in range(0, len(fen[i])):
        temp.append(X[fen[i][j]])
        temp1.extend(Y[fen[i][j]])
    X_part.append(temp)
    Y_part.append(temp1)

X_train = []
Y_train = []
X_test = X_part[2]
Y_test = Y_part[2]
for j in range(1, 5):
    X_train.extend(X_part[(2 + j) % 5])
    Y_train.extend(Y_part[(2 + j) % 5])

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(Y_train, dtype=torch.int64)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(Y_test, dtype=torch.int64)


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

net = Net(n_feature=32*32, n_hidden=10000, n_output=10) # 几个类别就几个 output
net.load_state_dict(torch.load('net2_par.pkl'))

optimizer = torch.optim.SGD(net.parameters(), lr=0.04)  # 传入 net 的所有参数, 学习率
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()



for t in range(350):
    out = net(X_train_tensor)  # 喂给 net 训练数据 x, 输出分析值
    loss = loss_func(out, Y_train_tensor)  # 计算两者的误差
    print(t, loss.data)
    optimizer.zero_grad()  # 清空上一步的残余更新参数值
    loss.backward()  # 误差反向传播, 计算参数更新值
    optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
torch.save(net.state_dict(), 'net2_par.pkl')   # parameters



out = net(X_test_tensor)
prediction = torch.max(F.softmax(out, dim=1), 1)[1]
pred_y = prediction.data.numpy().squeeze()
target_y = Y_test_tensor.data.numpy()
accuracy = sum(pred_y == target_y) / X_test_tensor.shape[0]  # 预测中有多少和真实值一样
print(accuracy)
print(target_y)
print('-'*50)
print(pred_y)
print(len(pred_y))
