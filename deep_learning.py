import numpy as np
import matplotlib.pyplot as plt
import time
m = 30
n0 = 1
n1 = 5
n2 = 1
lr2 = 0.01
lr1 = 0.01
# X = np.array([[-5, -7.5], [-3, -3.2], [-2, -1.2], [-1, 0.8], [0, 3.4], [2, 6.5], [3, 9.2], [7, 18]])    # 2x+3
# x_t = X[:, 0].reshape(m, 1)
# y_t = X[:, 1].reshape(m, 1)
x_t = np.linspace(-5, 5, m).reshape(m, 1)  # (7, 1)
y_t = np.exp(x_t) + 0.01*np.random.random(m).reshape(m, 1)  # (7, 1)

w1_t = np.random.randn(n1, n0) * 0.01  # (4, 1)
b1_t = np.random.randn(n1, 1)  # (4, 1)
w2_t = np.random.randn(n2, n1) * 0.01  # (1, 4)
b2_t = np.random.randn(1, 1)  # (1, 1)
x_test = np.linspace(-4, 4, 100)
plt.ion()   # something about plotting
def get_y_hat(x, w1, b1, w2, b2):
    z1 = np.dot(w1, x.T) + b1  # (4, 7)
    a1 = 1 / (1 + np.exp(-z1))  # (4, 7)
    z2 = np.dot(w2, a1) + b2  # (1, 7)
    a2 = 1 / (1 + np.exp(-z2))  # (1, 7)
    return a2*100, a1


def train(x, y, w1, b1, w2, b2):
    for i in range(300):
        a2, a1 = get_y_hat(x, w1, b1, w2, b2)
        j = np.sum((a2 - y.T) ** 2)/m  # (1, 1)
        print(j)
        dz2 = a2 - y.T  # (1, 7)
        dw2 = np.dot(dz2, a1.T) / m  # (1, 4)
        db2 = np.sum(dz2) / m  # (1, 1)
        da1 = np.dot(w2.T, dz2)
        dz1 = da1 * a1 * (1 - a1)
        dw1 = np.dot(dz1, x) / m
        db1 = np.sum(dz1) / m
        w2 = w2 - lr2 * dw2
        b2 = b2 - lr2 * db2
        w1 = w1 - lr1 * dw1
        b1 = b1 - lr1 * db1
        plt.cla()
        temp, y_test = get_y_hat(x_test.reshape(100, 1), w1, b1, w2, b2)
        plt.plot(x_t, y_t)
        plt.plot(x_test, temp.reshape(100, 1), color='red')
        plt.pause(0.1)

train(x_t, y_t, w1_t, b1_t, w2_t, b2_t)
plt.ioff()
plt.show()

