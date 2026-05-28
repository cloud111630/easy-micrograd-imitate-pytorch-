import matplotlib.pyplot as plt

#梯度下降

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 #权重

def forward(x):
    return x * w

def cost(xs, ys): #随机梯度只需求单个，随机梯度下面循环也只需拿一个样本进行更新，这里是求全部
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y_pred - y) ** 2
    return cost/len(xs)

def gradient(xs, ys):
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

print('predict (before training)', 4, forward(4))

epoch_list = []
cost_list = []

for epoch in range(100):
    cost_val = cost(x_data, y_data)
    grad_val = gradient(x_data, y_data)
    w -= 0.01 * grad_val #0.01为学习率

    epoch_list.append(epoch)
    cost_list.append(cost_val)

    print('Epoch:', epoch, 'w=', w, 'loss=', cost_val)

print('Predict (after training)', 4, forward(4))

plt.plot(epoch_list, cost_list)
plt.title('Cost in each epoch')
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()