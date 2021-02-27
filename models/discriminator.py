import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
                               kernel_size=3, stride=2, padding=1)  # out: 32 x 32 x 32
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 32 x 32 x 32
        self.b2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
                               stride=2, padding=1, bias=False)  # out: 32 x 32 x 32
        self.b3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=32,
                               kernel_size=3, stride=2, padding=1)  # out: 32 x 32 x 32
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=4*4*32, out_features=1)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.b2(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.b3(x)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.flatten(x)
        x = torch.sigmoid(self.fc1(x))

        return x
