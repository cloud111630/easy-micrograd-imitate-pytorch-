import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

def forward(x): #模型
    return x * w + b

def loss(x, y): #损失函数a
    y_pred = forward(x) #y_hat，预测
    return (y_pred - y) * (y_pred - y)

w_list = []
b_list = []
mse_list = []
for w in np.arange(0.0, 4.1, 0.1):
    for b in np.arange(-2.0, 2.1, 0.1):
        print('w=', w)
        print('b=', b)
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred_val = forward(x_val)
            loss_val = loss(x_val, y_val)
            l_sum += loss_val
            print('\t', x_val, y_val, y_pred_val, loss_val)

    print('MSE=', l_sum/3)
    w_list.append(w)
    b_list.append(b)
    mse_list.append(l_sum / 3)

plt.plot(w_list, mse_list)
plt.ylabel('Loss')
plt.xlabel('w')

plt.show()