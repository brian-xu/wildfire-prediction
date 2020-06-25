import torch
import torch.nn as nn
import torch.nn.functional as F


class FireCast(nn.Module):
    def __init__(self, area=15):
        super(FireCast, self).__init__()
        # self.avg_pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels=6, out_channels=32, kernel_size=2, stride=1),
                                   nn.Sigmoid(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Dropout2d())
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1),
                                   nn.ReLU(),
                                   nn.MaxPool2d(kernel_size=2, stride=2),
                                   nn.Dropout2d())

        # Calculates tensor size after a Conv2d
        def conv2d_result(size, kernel_size, stride=1):
            return (size - kernel_size) // stride + 1

        conv2d_size = conv2d_result(area, 2)
        conv2d_size = conv2d_result(conv2d_size, 2, 2)
        conv2d_size = conv2d_result(conv2d_size, 2)
        conv2d_size = conv2d_result(conv2d_size, 2, 2)
        self.linear_input_size = (conv2d_size ** 2) * 64

        # classifier
        self.dense1 = nn.Linear(self.linear_input_size, 128)
        self.dense2 = nn.Linear(133, 1)

    def forward(self, terrain, weather):
        # terrain = self.avg_pool1(terrain)
        x1 = self.conv1(terrain.float())
        x1 = self.conv2(x1)
        x1 = x1.view(-1, self.linear_input_size)
        x1 = self.dense1(x1)
        x1 = F.relu(x1)
        x2 = weather.float()

        x = torch.cat((x1, x2), dim=1)
        x = self.dense2(x)
        x = torch.sigmoid(x)
        return x
