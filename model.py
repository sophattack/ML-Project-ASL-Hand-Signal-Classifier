import torch.nn as nn
import torch
import torch.nn.functional as F


class CNN(nn.Module):

    def __init__(self, use_bn, num_fc1_neurons, num_conv_kernels):

        super(CNN, self).__init__()

        self.conv1 = nn.Conv2d(3, num_conv_kernels, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_conv_kernels, num_conv_kernels, 3)
        self.use_bn = use_bn
        self.num_conv_kernels = num_conv_kernels

        if use_bn:
            self.conv2_bn = nn.BatchNorm2d(num_conv_kernels)
            self.fc1_bn = nn.BatchNorm1d(num_fc1_neurons)
        self.fc1 = nn.Linear(num_conv_kernels*12*12, num_fc1_neurons)
        self.fc2 = nn.Linear(num_fc1_neurons, 10)

    def forward(self, features):
        x = self.pool(F.relu(self.conv1(features)))
        if self.use_bn:
            x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_conv_kernels*12*12)
        if self.use_bn:
            x = F.relu(self.fc1_bn(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x)

        return x


class CNN1layer(nn.Module):

    def __init__(self, use_bn, num_fc1_neurons, num_conv_kernels):

        super(CNN1layer, self).__init__()

        self.conv1 = nn.Conv2d(3, num_conv_kernels, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.use_bn = use_bn
        self.num_conv_kernels = num_conv_kernels

        if use_bn:
            self.conv2_bn = nn.BatchNorm2d(num_conv_kernels)
            self.fc1_bn = nn.BatchNorm1d(num_fc1_neurons)

        self.fc1 = nn.Linear(num_conv_kernels*27*27, num_fc1_neurons)
        self.fc2 = nn.Linear(num_fc1_neurons, 10)

    def forward(self, features):
        if self.use_bn:
            x = self.pool(F.relu(self.conv2_bn(self.conv1(features))))
        else:
            x = self.pool(F.relu(self.conv2(features)))
        x = x.view(-1, self.num_conv_kernels*27*27)
        if self.use_bn:
            x = F.relu(self.fc1_bn(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))

        x = F.relu(self.fc2(x))
        x = F.softmax(x)

        return x


class CNN2layer(nn.Module):

    def __init__(self, use_bn, num_fc1_neurons, num_conv_kernels):

        super(CNN2layer, self).__init__()

        self.conv1 = nn.Conv2d(3, num_conv_kernels, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_conv_kernels, num_conv_kernels, 3)
        self.use_bn = use_bn
        self.num_conv_kernels = num_conv_kernels

        if use_bn:
            self.conv2_bn = nn.BatchNorm2d(num_conv_kernels)
            self.fc1_bn = nn.BatchNorm1d(num_fc1_neurons)

        self.fc1 = nn.Linear(num_conv_kernels*12*12, num_fc1_neurons)
        self.fc2 = nn.Linear(num_fc1_neurons, 10)

    def forward(self, features):
        x = self.pool(F.relu(self.conv1(features)))
        if self.use_bn:
            x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.num_conv_kernels*12*12)
        if self.use_bn:
            x = F.relu(self.fc1_bn(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x)

        return x


class CNN4layer(nn.Module):

    def __init__(self, use_bn, num_fc1_neurons, num_conv_kernels):

        super(CNN4layer, self).__init__()

        self.conv1 = nn.Conv2d(3, num_conv_kernels, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(num_conv_kernels, num_conv_kernels, 3)
        self.use_bn = use_bn
        self.num_conv_kernels = num_conv_kernels
        self.conv3 = nn.Conv2d(num_conv_kernels, num_conv_kernels, 3)
        self.conv4 = nn.Conv2d(num_conv_kernels, num_conv_kernels, 3)

        if use_bn:
            self.conv2_bn = nn.BatchNorm2d(num_conv_kernels)
            self.fc1_bn = nn.BatchNorm1d(num_fc1_neurons)

        self.fc1 = nn.Linear(num_conv_kernels*1*1, num_fc1_neurons)
        self.fc2 = nn.Linear(num_fc1_neurons, 10)

    def forward(self, features):
        x = self.pool(F.relu(self.conv1(features)))
        if self.use_bn:
            x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = x.view(-1, self.num_conv_kernels*1*1)
        if self.use_bn:
            x = F.relu(self.fc1_bn(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x)

        return x


class BestModel(nn.Module):
    def __init__(self, use_bn):

        super(BestModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 10, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(10, 10, 7)
        self.use_bn = use_bn

        if use_bn:
            self.conv2_bn = nn.BatchNorm2d(10)
            self.fc1_bn = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(10*10*10, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, features):
        x = self.pool(F.relu(self.conv1(features)))
        if self.use_bn:
            x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 10*10*10)
        if self.use_bn:
            x = F.relu(self.fc1_bn(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x)

        return x


class BestSmallModel(nn.Module):

    def __init__(self, use_bn):

        super(BestSmallModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 5, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(5, 5, 9)
        self.use_bn = use_bn
        self.conv3 = nn.Conv2d(5, 5, 3)
        # self.conv4 = nn.Conv2d(30, 30, 3)

        if use_bn:
            self.conv2_bn = nn.BatchNorm2d(5)
            self.fc1_bn = nn.BatchNorm1d(32)

        self.fc1 = nn.Linear(5*3*3, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, features):
        x = self.pool(F.relu(self.conv1(features)))
        if self.use_bn:
            x = self.pool(F.relu(self.conv2_bn(self.conv2(x))))
        else:
            x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 5*3*3)
        if self.use_bn:
            x = F.relu(self.fc1_bn(self.fc1(x)))
        else:
            x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.softmax(x)

        return x