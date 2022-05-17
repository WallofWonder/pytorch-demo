"""
逻辑回归模型，多维特征(diabetes数据集)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class LogiRegDiabetesModel(nn.Module):
    def __init__(self):
        super(LogiRegDiabetesModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.sigmoid(self.linear3(x))
        return x


xy_data = np.loadtxt(fname='data/diabetes.csv', delimiter=',', dtype=np.float32, skiprows=1)
x_data = torch.from_numpy(xy_data[:, :-1])
y_data = torch.from_numpy(xy_data[:, [-1]])

losses = []
training_epochs = 1000
model = LogiRegDiabetesModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(training_epochs):
    y_pred = model(x_data)
    loss = F.binary_cross_entropy(y_pred, y_data, reduction='mean')

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(epoch, loss.item())
    losses.append(loss.item())

plt.figure()
plt.plot(range(training_epochs), losses)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.title('loss with epoch')
plt.show()
