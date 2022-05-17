"""
逻辑回归模型
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_data = torch.tensor([
    [1.],
    [2.],
    [3.]
])

y_data = torch.tensor([
    [0.],
    [0.],
    [1.]
])


class LogisticRegressionModel(nn.Module):
    def __init__(self):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        return y_pred


losses = []
epochs = 1000
model = LogisticRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(epochs):
    y_pred = model(x_data)
    loss = F.binary_cross_entropy(y_pred, y_data, reduction='sum')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, loss.item())
    losses.append(loss.item())

plt.figure()
plt.plot(range(epochs), losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss with epoch')
plt.show()

x = np.linspace(0, 10, 200)
x_t = torch.Tensor(x).view((200, 1))
y_t = model(x_t)
y = y_t.data.numpy()
plt.plot(x, y)
plt.plot([0, 10], [0.5, 0.5], color='red')
plt.grid()
plt.show()
