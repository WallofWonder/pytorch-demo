"""
逻辑回归模型，多维特征(diabetes数据集，使用DataLoader)
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn


class DiabetesDataSet(Dataset):
    def __init__(self, filepath):
        xy_data = np.loadtxt(filepath, delimiter=',', dtype=np.float32, skiprows=1)
        self.len = xy_data.shape[0]
        self.x_data = torch.from_numpy(xy_data[:, :-1])
        self.y_data = torch.from_numpy(xy_data[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


class LogiRegDiabetesModel(nn.Module):
    def __init__(self):
        super(LogiRegDiabetesModel, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)

    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.sigmoid(self.linear3(x))
        return x


def load_data(filepath, batch_size, use_cuda):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    return DataLoader(dataset=DiabetesDataSet(filepath=filepath),
                      batch_size=batch_size,
                      shuffle=True,
                      **kwargs)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using {}...".format("cuda" if use_cuda else "cpu"))

    losses = []
    training_epochs = 10
    batch_size = 64
    train_loader = load_data('../data/diabetes.csv', batch_size, use_cuda)
    model = LogiRegDiabetesModel().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(training_epochs):
        print(f'\n======= epoch {epoch} =========')
        running_loss = 0
        for index, (inputs, label) in enumerate(train_loader, 0):
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()
            label_pred = model(inputs)
            loss = F.binary_cross_entropy(label_pred, label, reduction='mean')
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if index % 2 == 1:
                running_loss /= 2.0
                losses.append(running_loss)
                print(f'batch: {index}, loss: {running_loss}')
                running_loss = 0

    plt.figure()
    plt.plot(range(len(losses)), losses)
    plt.ylabel('loss')
    plt.show()
