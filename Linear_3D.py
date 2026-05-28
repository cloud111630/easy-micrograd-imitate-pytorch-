import numpy as np
import matplotlib.pyplot as plt

#准备数据
x_data = np.array([1.0, 2.0, 3.0])
y_data = np.array([2.0, 4.0, 6.0])

#生成w & b的网格
w_vals = np.arange(0.0, 4.1, 0.1)
b_vals = np.arange(-2.0, 2.1, 0.1)

W, B = np.meshgrid(w_vals, b_vals) #将w_vals & b_vals转换为二维数组，形状(len(b_vals), len(w_vals))，通过影响方式确定axis[0]=b,axis[1]=w

X = x_data.reshape(1, 1, -1)
Y_true = y_data.reshape(1, 1, -1)

#广播
Y_pred = W[:, :, np.newaxis] * X + B[:, :, np.newaxis]
MSE = np.mean((Y_pred - Y_true) ** 2, axis=2)

#绘制3D
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
surf = ax.plot_surface(W, B, MSE, cmap='viridis')

ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('MSE')
ax.set_title('Loss Surface for y = w*x + b')

plt.show()