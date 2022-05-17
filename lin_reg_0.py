"""
线性回归模型
"""

import torch
import numpy as np
import matplotlib.pyplot as plt

w1 = torch.tensor([1.0])
w2 = torch.tensor([1.0])
w1.requires_grad = True
w2.requires_grad = True

x_data = np.linspace(1.0, 3.0, 3)
y_data = np.linspace(2.0, 6.0, 3)


def forward(x):
    return w1 * (x ** 2) + w2 * x


def loss(x, y):
    y_pred = forward(x)
    return (y_pred - y) ** 2


print(f'predict (before training) 8: {forward(4).item()}')

ls = []
for epoch in range(100):
    l = 0
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad w1:', x, y, w1.grad.item())
        print('\tgrad w2:', x, y, w2.grad.item())
        w1.data = w1.data - 0.01 * w1.grad.data
        w2.data = w2.data - 0.01 * w2.grad.data
        w1.grad.data.zero_()
    ls.append(l.item())
    print('progress:', epoch, l.item())

print(f'predict (after training) 8: {forward(8).item()}')

plt.figure()
plt.plot(range(100), ls)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss with epoch')
plt.show()

x = int(input('input x (-1 to quit) :'))
while x != -1:
    print(f'predict: {forward(x).item()}')
    x = int(input('input x (-1 to quit) :'))
