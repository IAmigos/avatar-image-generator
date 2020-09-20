# wandb login 17d2772d85cbda79162bd975e45fdfbf3bb18911

import argparse
from utils import parse_configuration

import math
import numpy as np
import random
import matplotlib.pyplot as plt
import os
import pathlib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
from torch.utils.data import DataLoader
from torch.autograd import Variable

from PIL import Image
import cv2

import time
from tqdm import tqdm
import helper
import wandb


def train(config_file, export=True):
    # WANDB configuration
    wandb.init(project="avatar_image_generator")
    wandb.watch_called = False
    config = wandb.config
    config.seed = 0
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    print('Reading config file...')
    configuration = parse_configuration(config_file)

    print('Initializing dataset...')

    path_faces = configuration['train_dataset_params']['loader_params']['dataset_path_faces']
    path_cartoons = configuration['train_dataset_params']['loader_params']['dataset_path_cartoons']
    config.image_size = configuration['train_dataset_params']['loader_params']['image_size']
    config.batch_size = configuration['train_dataset_params']['loader_params']['batch_size']
    config.workers = configuration['train_dataset_params']['loader_params']['num_workers']

    # Faces dataset

    transformFaces = T.Compose([
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor()
    ])

    dataset_faces = torchvision.datasets.ImageFolder(
        path_faces, transform=transformFaces)

    train_dataset_faces, test_dataset_faces = torch.utils.data.random_split(
        dataset_faces, (int(len(dataset_faces)*0.9), len(dataset_faces) - int(len(dataset_faces)*0.9)))

    train_loader_faces = torch.utils.data.DataLoader(
        train_dataset_faces,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers)

    test_loader_faces = torch.utils.data.DataLoader(
        test_dataset_faces,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers)

    transformCartoons = T.Compose([
        T.CenterCrop(300),
        T.Resize((config.image_size, config.image_size)),
        T.ToTensor(),
        # T.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
    ])

    # Cartoons dataset

    dataset_cartoons = torchvision.datasets.ImageFolder(
        path_cartoons, transform=transformCartoons)

    train_dataset_cartoons, test_dataset_cartoons = torch.utils.data.random_split(dataset_cartoons, (int(
        len(dataset_cartoons)*0.9), len(dataset_cartoons) - int(len(dataset_cartoons)*0.9)))

    train_loader_cartoons = torch.utils.data.DataLoader(
        train_dataset_cartoons,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers)

    test_loader_cartoons = torch.utils.data.DataLoader(
        test_dataset_cartoons,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers)

    # Hyperparameters

    num_epochs = configuration['model_params']['num_epochs']
    config.use_gpu = configuration['model_params']['use_gpu']
    lr_opXgan = configuration['model_params']['lr_opXgan']
    lr_opDisc = configuration['model_params']['lr_opDisc']
    b1_disc = configuration['model_params']['b1_disc']
    lr_opCdann = configuration['model_params']['lr_opCdann']
    b1_cdann = configuration['model_params']['b1_cdann']
    lr_denoiser = configuration['model_params']['lr_denoiser']
    wRec = configuration['model_params']['wRec']
    wClas = configuration['model_params']['wClas']
    wSem = configuration['model_params']['wSem']
    wGen = configuration['model_params']['wGen']


def save_weights(path, n):

    torch.save(e1.state_dict(), os.path.join(path, 'e1.pth'))
    torch.save(e2.state_dict(), os.path.join(path, 'e2.pth'))
    torch.save(e_shared.state_dict(), os.path.join(path, 'e_shared.pth'))
    torch.save(d_shared.state_dict(), os.path.join(path, 'd_shared.pth'))
    torch.save(d1.state_dict(), os.path.join(path, 'd1.pth'))
    torch.save(d2.state_dict(), os.path.join(path, 'd2.pth'))
    torch.save(c_dann.state_dict(), os.path.join(path, 'c_dann.pth'))
    torch.save(discriminator1.state_dict(), os.path.join(path, 'disc1.pth'))
    torch.save(denoiser.state_dict(), os.path.join(path, 'denoiser.pth'))

    wandb.save(os.path.join(path, 'e1.pth'))
    wandb.save(os.path.join(path, 'e2.pth'))
    wandb.save(os.path.join(path, 'e_shared.pth'))
    wandb.save(os.path.join(path, 'd_shared.pth'))
    wandb.save(os.path.join(path, 'd1.pth'))
    wandb.save(os.path.join(path, 'd2.pth'))
    wandb.save(os.path.join(path, 'c_dann.pth'))
    wandb.save(os.path.join(path, 'disc1.pth'))
    wandb.save(os.path.join(path, 'denoiser.pth'))
