#import wandb
import math
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

#from tqdm.notebook import tqdm
from PIL import Image

import logging

from keras_segmentation.pretrained import pspnet_101_voc12
import cv2
#import helper
import json


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


class Dshared(nn.Module):
    def __init__(self):
        super(Dshared, self).__init__()
        #c = capacity
        self.deconv1 = nn.ConvTranspose2d(
            in_channels=1024, out_channels=512, kernel_size=4, stride=2, bias=False)
        self.bd1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(
            in_channels=512, out_channels=256, kernel_size=2, stride=2, bias=False)
        self.bd2 = nn.BatchNorm2d(256)

        nn.init.kaiming_normal_(self.deconv1.weight)
        nn.init.kaiming_normal_(self.deconv2.weight)

    def forward(self, x):
        x = x.view(-1, 1024, 1, 1)
        x = F.relu(self.deconv1(x))
        x = self.bd1(x)
        x = F.relu(self.deconv2(x))
        x = self.bd2(x)

        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.deconv3 = nn.ConvTranspose2d(
            in_channels=256, out_channels=128, kernel_size=2, stride=2, bias=False)
        self.bd3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(
            in_channels=128, out_channels=64, kernel_size=2, stride=2, bias=False)
        self.bd4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=2, stride=2)

        nn.init.kaiming_normal_(self.deconv3.weight)
        nn.init.kaiming_normal_(self.deconv4.weight)
        nn.init.xavier_normal_(self.deconv5.weight)

    def forward(self, x):
        x = F.relu(self.deconv3(x))
        x = self.bd3(x)
        x = F.relu(self.deconv4(x))
        x = self.bd4(x)
        x = torch.tanh(self.deconv5(x))

        return x


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse.apply(x)


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


class Avatar_Generator_Model():
    """
    # Methods
    __init__(dict_model): initializer
    dict_model: layers required to perform face-to-image generation (e1, e_shared, d_shared, d2, denoiser)
    generate(face_image, output_path=None): reutrn cartoon generated from given face image, saves it to output path if given
    load_weights(weights_path): loads weights from given path
    """

    def __init__(self, weights_path, device):
        self.segmentation = pspnet_101_voc12()
        self.e1 = Encoder()
        self.e_shared = Eshared()
        self.d_shared = Dshared()
        self.d2 = Decoder()
        self.denoiser = Denoiser()
        self.load_weights(weights_path, device)

    def generate(self, path_filename, output_path):
        face = self.__extract_face(path_filename, output_path)
        return self.__to_cartoon(face, output_path)

    def load_weights(self, weights_path, device):

        self.e1.load_state_dict(torch.load(
            weights_path + 'e1.pth', map_location=torch.device(device)))

        self.e_shared.load_state_dict(
            torch.load(weights_path + 'e_shared.pth', map_location=torch.device(device)))

        self.d_shared.load_state_dict(
            torch.load(weights_path + 'd_shared.pth', map_location=torch.device(device)))

        self.d2.load_state_dict(torch.load(
            weights_path + 'd2.pth', map_location=torch.device(device)))

        self.denoiser.load_state_dict(
            torch.load(weights_path + 'denoiser.pth', map_location=torch.device(device)))

    def __extract_face(self, path_filename, output_path, extension=""):
        #import model
        # segment image
        # remove background
        # return face

        #output_file = path_filename.split('/')[-1].split('.')[0] + extension + ".jpg"

        out = self.segmentation.predict_segmentation(
            inp=path_filename,
            out_fname=output_path
        )

        img_mask = cv2.imread(output_path)
        img1 = cv2.imread(path_filename)  # READ BGR

        seg_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
        _, bg_mask = cv2.threshold(
            seg_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

        bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

        bg = cv2.bitwise_or(img1, bg_mask)

        cv2.imwrite(output_path, bg)
        face_image = Image.open(output_path)

        return face_image

    def __to_cartoon(self, face, output_path):
        self.e1.eval()
        self.e_shared.eval()
        self.d_shared.eval()
        self.d2.eval()
        self.denoiser.eval()

        transform = transforms.Compose(
            [transforms.Resize((64, 64)), transforms.ToTensor()])
        face = transform(face).float()
        X = face.unsqueeze(0)
        with torch.no_grad():
            output = self.e1(X)
            output = self.e_shared(output)
            output = self.d_shared(output)
            output = self.d2(output)
            output = self.denoiser(output)

        output = output[0]

        # if output_path is not None:
        # save to path
        # fileName.jpg part of output_path
        torchvision.utils.save_image(tensor=output, fp=output_path)

        return (torchvision.transforms.ToPILImage()(output), output)


class Critic(nn.Module):
    '''
    Critic Class
    Values:
        im_chan: the number of channels in the images, fitted for the dataset used, a scalar
              (MNIST is black-and-white, so 1 channel is your default)
        hidden_dim: the inner dimension, a scalar
    '''

    def __init__(self, im_chan=1, hidden_dim=1024):
        super(Critic, self).__init__()
        self.crit = nn.Sequential(
            self.make_crit_block(im_chan, hidden_dim),
            self.make_crit_block(hidden_dim, hidden_dim * 2),
            self.make_crit_block(hidden_dim * 2, 1, final_layer=True),
        )

    def make_crit_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        '''
        Function to return a sequence of operations corresponding to a critic block of DCGAN;
        a convolution, a batchnorm (except in the final layer), and an activation (except in the final layer).
        Parameters:
            input_channels: how many channels the input feature representation has
            output_channels: how many channels the output feature representation should have
            kernel_size: the size of each convolutional filter, equivalent to (kernel_size, kernel_size)
            stride: the stride of the convolution
            final_layer: a boolean, true if it is the final layer and false otherwise 
                      (affects activation and batchnorm)
        '''
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels,
                          kernel_size, stride),
            )

    def forward(self, image):
        '''
        Function for completing a forward pass of the critic: Given an image tensor, 
        returns a 1-dimension tensor representing fake/real.
        Parameters:
            image: a flattened image tensor with dimension (im_chan)
        '''
        image = grad_reverse(image)
        crit_pred = self.crit(image)
        return crit_pred.view(len(crit_pred), -1)


def get_gradient(crit, real, fake, epsilon):
    '''
    Return the gradient of the critic's scores with respect to mixes of real and fake images.
    Parameters:
        crit: the critic model
        real: a batch of real images
        fake: a batch of fake images
        epsilon: a vector of the uniformly random proportions of real/fake per mixed image
    Returns:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    '''
    # Mix the images together
    mixed_images = real * epsilon + fake * (1 - epsilon)

    # Calculate the critic's scores on the mixed images
    mixed_scores = crit(mixed_images)

    # Take the gradient of the scores with respect to the images
    gradient = torch.autograd.grad(
        # Note: You need to take the gradient of outputs with respect to inputs.
        # This documentation may be useful, but it should not be necessary:
        # https://pytorch.org/docs/stable/autograd.html#torch.autograd.grad
        #### START CODE HERE ####
        inputs=mixed_images,
        outputs=mixed_scores,
        #### END CODE HERE ####
        # These other parameters have to do with the pytorch autograd engine works
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return gradient


def gradient_penalty(gradient):
    '''
    Return the gradient penalty, given a gradient.
    Given a batch of image gradients, you calculate the magnitude of each image's gradient
    and penalize the mean quadratic distance of each magnitude to 1.
    Parameters:
        gradient: the gradient of the critic's scores, with respect to the mixed image
    Returns:
        penalty: the gradient penalty
    '''
    # Flatten the gradients so that each row captures one image
    gradient = gradient.view(len(gradient), -1)

    # Calculate the magnitude of every row
    gradient_norm = gradient.norm(2, dim=1)

    # Penalize the mean squared distance of the gradient norms from 1
    #### START CODE HERE ####
    penalty = ((gradient_norm - 1)**2).mean()
    #### END CODE HERE ####
    return penalty


def get_crit_loss(crit_fake_pred, crit_real_pred, gp, c_lambda):
    '''
    Return the loss of a critic given the critic's scores for fake and real images,
    the gradient penalty, and gradient penalty weight.
    Parameters:
        crit_fake_pred: the critic's scores of the fake images
        crit_real_pred: the critic's scores of the real images
        gp: the unweighted gradient penalty
        c_lambda: the current weight of the gradient penalty 
    Returns:
        crit_loss: a scalar for the critic's loss, accounting for the relevant factors
    '''
    #### START CODE HERE ####
    crit_loss = -(crit_real_pred - crit_fake_pred - c_lambda*gp).mean()
    #### END CODE HERE ####
    return crit_loss
