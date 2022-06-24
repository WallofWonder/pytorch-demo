"""
Titanic 数据集预测
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from torch import optim
from torch.utils.data import Dataset, DataLoader
import logging


class Titanic(nn.Module):
    def __init__(self):
        super(Titanic, self).__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 2)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class TitanicDataset(Dataset):
    def __init__(self, filepath):
        data_csv = pd.read_csv(filepath)
        # Sex
        # 编码Sex为二进制0 1
        sex_bin = LabelBinarizer().fit(data_csv.loc[:, 'Sex'])
        # 将源数据映射为0 1
        data_csv['Sex'] = sex_bin.transform(data_csv['Sex'])

        # Embarked
        # 将nan值补为S
        data_csv['Embarked'] = data_csv['Embarked'].fillna('S')
        # 转换为one-hot
        embarked_onehot = pd.get_dummies(data_csv['Embarked'])
        # 添加到原数据
        data_csv = pd.concat([data_csv, embarked_onehot], axis=1)

        # 添加特征Family，统一SibSp和Parch
        data_csv['Family'] = data_csv['SibSp'] + data_csv['Parch'] + 1
        # 添加特征Alone，如果没有SibSp和Parch，则是单人登船
        data_csv['Alone'] = data_csv['Family'].apply(lambda x: 0 if x > 1 else 1)

        # Age
        data_csv['Age'] = data_csv['Age'].fillna(-0.5)

        # 选择以下特征进行学习
        features = ['Pclass', 'Sex', 'Age', 'C', 'Q', 'S', 'Alone']
        self.x_data = data_csv[features].values
        self.y_data = data_csv['Survived'].values
        self.y_data = pd.get_dummies(self.y_data).to_numpy()

        # 特征缩放
        sc = StandardScaler()
        self.x_data = sc.fit_transform(self.x_data)
        # 转为Torch.tensor
        self.x_data = torch.from_numpy(self.x_data).to(torch.float32)
        self.y_data = torch.from_numpy(self.y_data.reshape(891, 2)).to(torch.float32)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return len(self.x_data)


def load_data(filepath, batch_size, use_cuda):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    return DataLoader(dataset=TitanicDataset(filepath=filepath),
                      batch_size=batch_size,
                      shuffle=True,
                      **kwargs)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using {}...".format("cuda" if use_cuda else "cpu"))

    # 创建logger对象
    logger = logging.getLogger('train_logger')
    # 设置日志等级
    logger.setLevel(logging.DEBUG)
    train_log = logging.FileHandler('train.log', 'a', encoding='utf-8')
    # 向文件输出的日志级别
    train_log.setLevel(logging.DEBUG)
    # 向文件输出的日志信息格式
    train_log.setFormatter(logging.Formatter(
        '%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s'))
    # 加载文件到logger对象中
    logger.addHandler(train_log)
    log_interval = 10

    losses = []
    train_loss = 0
    train_loss_min = np.Inf
    accuracies = []
    training_epochs = 200
    batch_size = 64
    train_loader = load_data('../../data/titanic/train.csv', batch_size, use_cuda)
    data_size = len(train_loader.dataset)
    model = Titanic().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(training_epochs):
        running_loss = 0
        correct_num = 0
        for index, (inputs, label) in enumerate(train_loader, 0):
            inputs, label = inputs.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(inputs)
            loss = F.cross_entropy(output, label)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            predict = output.argmax(dim=1, keepdim=True)
            actual = label.argmax(dim=1, keepdim=True)
            correct_num += predict.eq(actual).sum().item()

        train_loss = 1.0 * running_loss / data_size
        accuracy = correct_num / data_size
        losses.append(train_loss)
        accuracies.append(accuracy)

        if train_loss <= train_loss_min:
            torch.save(model.state_dict(), 'model.pt')
            train_loss_min = train_loss

        if (epoch + 1) % log_interval == 0:
            msg = 'epoch: {}\tloss: {:.3f}\tcorrect: {}/{}\taccuracy: {:.2f}%'.format(
                epoch, loss, correct_num, data_size, 100.0 * accuracy
            )
            logger.info(msg)
            print(msg)

    plt.figure()
    plt.plot(range(len(losses)), losses, color='blue')
    plt.title('Train Loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.plot(range(len(accuracies)), accuracies, color='red')
    plt.title('Train Accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.show()
