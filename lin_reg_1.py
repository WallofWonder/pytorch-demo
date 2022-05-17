"""
使用nn.Module搭建线性回归模型
"""

import matplotlib.pyplot as plt
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
    [2.],
    [4.],
    [6.]
])


class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_features=1, out_features=1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


losses = []
epochs = 1000
model = LinearModel()
optimizer = optim.SGD(model.parameters(), lr=0.01)

for epoch in range(epochs):
    y_pred = model(x_data)
    loss = F.mse_loss(y_pred, y_data, reduction='sum')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, loss.item())
    losses.append(loss.item())

print('w = ', model.linear.weight.item())
print('b = ', model.linear.bias.item())

x_test = torch.tensor([[4.]])
y_test = model(x_test)
print('y_pred = ', y_test.data)

plt.figure()
plt.plot(range(epochs), losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss with epoch')
plt.show()
