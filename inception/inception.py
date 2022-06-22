import torch
from torch import nn
import torch.nn.functional as F


class Inception(nn.Module):
    def __init__(self, in_channels):
        super(Inception, self).__init__()
        self.branch_pool = nn.Conv2d(in_channels=in_channels, out_channels=24, kernel_size=1)

        self.branch1x1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels=in_channels, out_channels=16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, padding=1)

    def forward(self, x):
        # 池化分支 
        # -> Avg_pool -> Conv_1x1(24) ->
        branch_pool = F.avg_pool2d(x, 3, 1, 1)
        branch_pool = self.branch_pool(branch_pool)

        # Conv_1x1 分支
        # -> Conv_1x1(16) ->
        branch1x1 = self.branch1x1(x)

        # Conv_5x5 分支 
        # -> Conv_1x1(16) -> Conv_5x5(24) ->
        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        # Conv_3x3 分支
        # -> Conv_1x1(16) -> Conv_3x3(24) -> Conv_3x3(24) ->
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        return torch.cat([branch_pool, branch1x1, branch5x5, branch3x3], dim=1)
