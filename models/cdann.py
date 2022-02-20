import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)

"""
class Cdann(nn.Module):
    def __init__(self, dropout_rate):
        super(Cdann, self).__init__()
        self.fc1 = nn.Linear(in_features=1024, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(in_features=256, out_features=128)
        self.fc4 = nn.Linear(in_features=128, out_features=64)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(in_features=64, out_features=32)
        self.fc6 = nn.Linear(in_features=32, out_features=16)
        self.dropout6 = nn.Dropout(dropout_rate)
        self.fc7 = nn.Linear(in_features=16, out_features=1)

        nn.init.kaiming_normal_(self.fc1.weight)
        nn.init.kaiming_normal_(self.fc2.weight)
        nn.init.kaiming_normal_(self.fc3.weight)
        nn.init.kaiming_normal_(self.fc4.weight)
        nn.init.kaiming_normal_(self.fc5.weight)
        nn.init.kaiming_normal_(self.fc6.weight)
        nn.init.xavier_normal_(self.fc7.weight)

    def forward(self, x):
        x = grad_reverse(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.dropout4(x)
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = self.dropout6(x)
        x = torch.sigmoid(self.fc7(x))

        return x
"""


        
class Cdann(nn.Module):
    '''
    Taken from Coursera - GANNs
    '''

    def __init__(self, use_critic_dann, im_chan=1024, hidden_dim=512, use_spectral_norm=False):
        super(Cdann, self).__init__()
        self.use_critic_dann = use_critic_dann
        self.cdan = nn.Sequential(
            self.make_cdan_block(im_chan, hidden_dim, use_spectral_norm),
            self.make_cdan_block(hidden_dim, hidden_dim // 2, use_spectral_norm),
            self.make_cdan_block(hidden_dim // 2, hidden_dim // 4, use_spectral_norm),
            self.make_cdan_block(hidden_dim // 4, 1, use_spectral_norm, final_layer=True),  
        )
        self.cdan = self.cdan.apply(self.weights_init)

    def make_cdan_block(self, input_channels, output_channels, use_spectral_norm, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                self.make_linear_block(input_channels, output_channels, use_spectral_norm),
                nn.BatchNorm1d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                self.make_linear_block(input_channels, output_channels, use_spectral_norm)
            )
    
    def make_linear_block(self, input_channels, output_channels, use_spectral_norm):
        if use_spectral_norm and self.use_critic_dann:
            return nn.Sequential(
                nn.utils.spectral_norm(nn.Linear(in_features=input_channels,
                            out_features=output_channels))
            )
            
        else:
            return nn.Sequential(
                nn.Linear(in_features=input_channels,
                            out_features=output_channels)
            )

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)
        if isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)

    def forward(self, feature):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            feature: a tensor with dimension (im_chan)
        '''
        feature = grad_reverse(feature)
        cdan_pred = self.cdan(feature)
        return cdan_pred.view(len(cdan_pred), -1)



