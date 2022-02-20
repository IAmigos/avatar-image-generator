import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader


class Discriminator(nn.Module):
    def __init__(self, use_critic_disc, use_spectral_norm):
        super(Discriminator, self).__init__()
        self.use_critic_disc = use_critic_disc
        self.conv1 = self.make_block(in_channels=3, 
                                    out_channels=16,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=True,
                                    use_spectral_norm=use_spectral_norm)

        self.conv2 = self.make_block(in_channels=16, 
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                    use_spectral_norm=use_spectral_norm)

        self.conv3 = self.make_block(in_channels=32, 
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=False,
                                    use_spectral_norm=use_spectral_norm)

        self.conv4 = self.make_block(in_channels=32, 
                                    out_channels=32,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1,
                                    bias=True,
                                    use_spectral_norm=use_spectral_norm)

        self.fc1 = self.make_block(in_channels=4*4*32,
                                    out_channels=1,
                                    use_spectral_norm=use_spectral_norm,
                                    final_layer=True)

        #self.conv1 = nn.Conv2d(in_channels=3, out_channels=16,
         #                      kernel_size=3, stride=2, padding=1)  # out: 32 x 32 x 32
        #self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
         #                      stride=2, padding=1, bias=False)  # out: 32 x 32 x 32
        self.b2 = nn.BatchNorm2d(32)
        #self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3,
         #                      stride=2, padding=1, bias=False)  # out: 32 x 32 x 32
        self.b3 = nn.BatchNorm2d(32)
        #self.conv4 = nn.Conv2d(in_channels=32, out_channels=32,
          #                     kernel_size=3, stride=2, padding=1)  # out: 32 x 32 x 32
        self.flatten = nn.Flatten()
        #self.fc1 = nn.Linear(in_features=4*4*32, out_features=1)

        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.conv3.weight)
        nn.init.kaiming_normal_(self.conv4.weight)
        nn.init.xavier_normal_(self.fc1.weight)

    def make_block(self, in_channels=None, out_channels=None, 
                    kernel_size=None, stride=None, padding=None,
                    use_spectral_norm=None, bias=True, final_layer=False):
        if not final_layer:
            if use_spectral_norm and self.use_critic_disc:
                return nn.utils.spectral_norm(
                            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias)
                                )
            else:
                return nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=bias)
        else:
            if use_spectral_norm and self.use_critic_disc:
                return nn.utils.spectral_norm(nn.Linear(in_features=in_channels, 
                                out_features=out_channels))
            else:
                return nn.Linear(in_features=in_channels, 
                                out_features=out_channels)
                                


    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.b2(x)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = self.b3(x)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = self.flatten(x)
        x = self.fc1(x)

        return x
