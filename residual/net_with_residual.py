import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import transforms, datasets

from residualblock import ResidualBlock


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5)
        self.mp = nn.MaxPool2d(kernel_size=2)

        self.rblock1 = ResidualBlock(channels=16)
        self.rblock2 = ResidualBlock(channels=32)

        self.fc = nn.Linear(in_features=512, out_features=10)

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x


n_epochs = 10
batch_size_train = 64
batch_size_test = 64
log_interval = 10
learning_rate = 0.01
momentum = 0.5
train_losses = []
train_counter = []
test_losses = []
test_counter = []
accuracies = []


# 数据加载
def data_loader(batch_size, batch_size_test, use_cuda=False):
    """
    数据加载器

    :param batch_size: 训练集批次大小
    :param batch_size_test:  测试集批次大小
    :param use_cuda: 是否使用GPU
    :return: 训练集和测试集
    """

    # GPU训练需要的参数
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}

    # 数据处理器
    transform = transforms.Compose([
        # 把[0,255]的(H,W,C)的图片转换为[0,1]的(channel,height,width)的图片
        transforms.ToTensor(),
        # z-score标准化为标准正态分布
        # 这两个数分别是MNIST的均值和标准差
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../data/',
                       train=True,
                       download=True,
                       transform=transform),
        batch_size=batch_size,
        shuffle=True,
        **kwargs)

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(root='../data/',
                       train=False,
                       transform=transform),
        batch_size=batch_size_test,
        shuffle=True,
        **kwargs)

    return train_loader, test_loader


# 训练脚本
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} ({:.0f}%)\tLoss: {:.6f}'.format(
                epoch, 100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * batch_size_train) + ((epoch - 1) * len(train_loader.dataset)))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():  # 预测时不需要反向传播
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            predict = output.argmax(dim=1, keepdim=True)
            correct += predict.eq(target.view_as(predict)).sum().item()

    # 上面test_loss得到的是累加和，这里求得均值
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    accuracies.append(100. * correct / len(test_loader.dataset))


def drawFig():
    """
    绘图

    :return:
    """
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(train_counter[:len(train_losses)], train_losses, color='blue')
    plt.scatter(test_counter[:len(test_losses)], test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.title('Loss on the training tata')
    plt.xlabel('number of training examples')
    plt.ylabel('loss')
    plt.show()

    plt.figure()
    plt.plot(range(len(accuracies) - 1), accuracies[1:])
    plt.title('Accuracy(%) on the test data')
    plt.xlabel('epoch of test')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == '__main__':
    # 启用英伟达cuDNN加速框架和CUDA
    torch.backends.cudnn.enabled = True
    torch.manual_seed(1)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using {}...".format("cuda" if use_cuda else "cpu"))

    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

    train_loader, test_loader = data_loader(batch_size=batch_size_train, batch_size_test=batch_size_test,
                                            use_cuda=use_cuda)
    test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

    test()
    for epoch in range(1, n_epochs + 1):
        train(epoch)
        test()

    drawFig()
