import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 32 x 32 x 32
        self.b1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 16 x 16 x 64
        self.b2 = nn.BatchNorm2d(64)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.b1(x)
        x = F.relu(self.conv2(x))
        x = self.b2(x)

        return x


class Eshared(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(Eshared, self).__init__()
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.b3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256,
                               kernel_size=3, stride=2, padding=1, bias=False)
        self.b4 = nn.BatchNorm2d(256)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4*4*256,
                             out_features=1024, bias=False)
        self.bfc1 = nn.BatchNorm1d(1024)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(in_features=1024, out_features=1024, bias=False)
        self.bfc2 = nn.BatchNorm1d(1024)

        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)

    def forward(self, x):
        x = F.relu(self.conv3(x))
        x = self.b3(x)
        x = F.relu(self.conv4(x))
        x = self.b4(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.bfc1(x)
        x = self.dropout2(x)
        x = F.relu(self.fc2(x))
        x = self.bfc2(x)

        return x