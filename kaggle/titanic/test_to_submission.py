import pandas as pd
import torch
from sklearn.preprocessing import LabelBinarizer, StandardScaler
from torch.utils.data import Dataset, DataLoader

from kaggle.titanic.titanic import Titanic


class TestDataset(Dataset):
    """
    Titanic测试数据
    """

    def __init__(self, filepath):
        data_csv = pd.read_csv(filepath)
        # Sex
        # 编码Sex为二进制0 1
        sex_bin = LabelBinarizer().fit(data_csv.loc[:, 'Sex'])
        # 将源数据映射为0 1
        data_csv['Sex'] = sex_bin.transform(data_csv['Sex'])

        # Embarked
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

        # 选择以下特征进行测试
        features = ['Pclass', 'Sex', 'Age', 'C', 'Q', 'S', 'Alone']
        self.x_data = data_csv[features].values
        self.passengerIds = data_csv['PassengerId'].values

        # 特征缩放
        sc = StandardScaler()
        self.x_data = sc.fit_transform(self.x_data)
        # 转为Torch.tensor
        self.x_data = torch.from_numpy(self.x_data).to(torch.float32)

    def __getitem__(self, index):
        return self.passengerIds[index], self.x_data[index]

    def __len__(self):
        return len(self.x_data)


def load_test_data(filepath, use_cuda):
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    return DataLoader(dataset=TestDataset(filepath=filepath),
                      shuffle=False,
                      **kwargs)


if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using {}...".format("cuda" if use_cuda else "cpu"))

    test_loader = load_test_data('../../data/titanic/test.csv', use_cuda=use_cuda)
    test_size = len(test_loader.dataset)
    model = Titanic().to(device)
    model.load_state_dict(torch.load('model.pt', map_location=torch.device(device)))

    model.eval()
    test_loss = 0
    correct = 0
    test_submission = []
    for index, (passengerId, input) in enumerate(test_loader, 0):
        passengerId, input = passengerId.to(device), input.to(device)
        output = model(input)
        output = output.argmax(dim=1)
        # passenger id, input 转化为基本int 类型存到csv
        passengerId = passengerId.cpu().numpy()[0]
        output = output.cpu().numpy()[0]
        test_submission.append([passengerId, output])

    df = pd.DataFrame(data=test_submission, columns=['PassengerId', 'Survived'])
    df.to_csv('submission.csv', index=False)
