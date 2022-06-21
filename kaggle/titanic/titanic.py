"""
Titanic 数据集预测
"""

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelBinarizer, StandardScaler


class Titanic(nn.Module):
    def __init__(self):
        super(Titanic, self).__init__()
        self.fc1 = nn.Linear(7, 512)
        self.fc2 = nn.Linear(522, 512)
        self.fc3 = nn.Linear(52, 2)
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

        # 特征缩放
        sc = StandardScaler()
        self.x_data = sc.fit_transform(self.x_data)
        # 转为Torch.tensor
        self.x_data = torch.from_numpy(self.x_data)
        self.y_data = torch.from_numpy(self.y_data.reshape(891, 1))

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]


data = TitanicDataset('../../data/titanic/train.csv')

model = Titanic()
