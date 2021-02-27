import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Denoiser(nn.Module):
    def __init__(self):
        super(Denoiser, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2))

        self.decoder = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(64, 3, kernel_size=3, padding=1))

    def forward(self, x):
        return torch.tanh(self.decoder(self.encoder(x)))