import numpy as np
import matplotlib.pyplot as plt

x_data = np.linspace(1.0, 3.0, 3)
y_data = np.linspace(2.0, 6.0, 3)
ws = np.arange(0., 4.1, .1)
bs = np.arange(-2., 2.1, .1)
W, B = np.meshgrid(ws, bs)
MSEs = []


def forward(x):
    print(f"w: {w}\tb: {b}")
    return x * w + b


def loss(y):
    return (y_pred - y) ** 2


for w in ws:
    for b in bs:
        l_sum = 0
        for x_val, y_val in zip(x_data, y_data):
            y_pred = forward(x_val)
            l_val = loss(y_val)
            l_sum += l_val
        mse = l_sum / len(x_data)
        MSEs.append(mse)
MSEs = np.array(MSEs).reshape(41, 41)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(W, B, MSEs, cmap='viridis', edgecolor='none')
ax.set_xlabel('W')
ax.set_ylabel('B')
ax.set_zlabel('MSE')
ax.set_title('MSE of wx+b')
plt.show()
