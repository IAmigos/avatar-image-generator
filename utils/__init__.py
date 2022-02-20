import wandb
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm
from PIL import Image

import logging
import random

from keras_segmentation.pretrained import pspnet_50_ADE_20K, pspnet_101_cityscapes, pspnet_101_voc12
import cv2
import helper
import json


IMAGE_SIZE = 64
MEAN = 0.5
SD = 0.5
STATS = (MEAN, MEAN, MEAN), (SD, SD, SD)

def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def parse_arguments():
    ap = argparse.ArgumentParser()
    ap.add_argument('-w', '--wandb', default=False, action='store_true',
                    help="use weights and biases")
    ap.add_argument('-nw  ', '--no-wandb', dest='wandb', action='store_false',
                    help="not use weights and biases")
    ap.add_argument('-n', '--run_name', required=False, type=str, default=None,
                    help="name of the execution to save in wandb")
    ap.add_argument('-nt', '--run_notes', required=False, type=str, default=None,
                    help="notes of the execution to save in wandb")

    args = ap.parse_args()

    return args


def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file, 'r') as json_file:
            return json.load(json_file)
    else:
        return config_file


def init_logger(log_file=None, log_dir=None):

    fmt = '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s: %(message)s'

    if log_dir is None:
        log_dir = '~/temp/log/'

    if not os.path.exists(log_dir):
        print("Creating dir")
        os.makedirs(log_dir)

    log_file = os.path.join(log_dir, log_file)

    print('log file path:' + log_file)

    logging.basicConfig(level=logging.INFO,
                        filename=log_file,
                        format=fmt)

    return logging


def configure_model(config_file, use_wandb):

    config_file = parse_configuration(config_file)

    config = dict(
        model_path=config_file["server_config"]["model_path"],
        download_directory=config_file["server_config"]["download_directory"],

        root_path=config_file["train_dataset_params"]["root_path"],
        dataset_path_faces=config_file["train_dataset_params"]["dataset_path_faces"],
        dataset_path_cartoons=config_file["train_dataset_params"]["dataset_path_cartoons"],
        dataset_path_test_faces=config_file["train_dataset_params"]["dataset_path_test_faces"],
        dataset_path_segmented_faces=config_file["train_dataset_params"]["dataset_path_segmented_faces"],
        dataset_path_output_faces=config_file["train_dataset_params"]["dataset_path_output_faces"],
        batch_size=config_file["train_dataset_params"]["loader_params"]["batch_size"],

        save_weights=config_file["train_dataset_params"]["save_weights"],
        num_backups=config_file["train_dataset_params"]["num_backups"],
        save_path=config_file["train_dataset_params"]["save_path"],

        dropout_rate_eshared=config_file["model_hparams"]["dropout_rate_eshared"],
        use_critic_dann=config_file["model_hparams"]["use_critic_dann"],
        use_critic_disc=config_file["model_hparams"]["use_critic_disc"],
        use_spectral_norm=config_file["model_hparams"]["use_spectral_norm"],
        use_denoiser=config_file["model_hparams"]["use_denoiser"],
        num_epochs=config_file["model_hparams"]["num_epochs"],
        learning_rate_opTotal=config_file["model_hparams"]["learning_rate_opTotal"],
        learning_rate_opDisc=config_file["model_hparams"]["learning_rate_opDisc"],
        learning_rate_denoiser=config_file["model_hparams"]["learning_rate_denoiser"],
        learning_rate_opCdann=config_file["model_hparams"]["learning_rate_opCdann"],
        wRec_loss=config_file["model_hparams"]["wRec_loss"],
        wDann_loss=config_file["model_hparams"]["wDann_loss"],
        wSem_loss=config_file["model_hparams"]["wSem_loss"],
        wGan_loss=config_file["model_hparams"]["wGan_loss"],
        wTeach_loss=config_file["model_hparams"]["wTeach_loss"],
        use_gpu=config_file["model_hparams"]["use_gpu"]
    )

    if not use_wandb:
        config = type("configuration", (object,), config)

    return config


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_uniform_(m.weight.data)

    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def save_weights(model, path_sub, use_wandb=True):
    e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser = model

    torch.save(e1.state_dict(), os.path.join(path_sub, 'e1.pth'))
    torch.save(e2.state_dict(), os.path.join(path_sub, 'e2.pth'))
    torch.save(e_shared.state_dict(), os.path.join(path_sub, 'e_shared.pth'))
    torch.save(d_shared.state_dict(), os.path.join(path_sub, 'd_shared.pth'))
    torch.save(d1.state_dict(), os.path.join(path_sub, 'd1.pth'))
    torch.save(d2.state_dict(), os.path.join(path_sub, 'd2.pth'))
    torch.save(c_dann.state_dict(), os.path.join(path_sub, 'c_dann.pth'))
    torch.save(discriminator1.state_dict(),
               os.path.join(path_sub, 'disc1.pth'))
    torch.save(denoiser.state_dict(), os.path.join(path_sub, 'denoiser.pth'))

    if use_wandb:
        wandb.save(os.path.join(path_sub, '*.pth'),
                   base_path='/'.join(path_sub.split('/')[:-2]))


def get_transforms_config_face():
    list_transforms = [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(*STATS)
    ]

    return list_transforms


def get_transforms_config_cartoon():
    list_transforms = [
        transforms.CenterCrop(400),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(*STATS)
    ]

    return list_transforms

def get_datasets(root_path, dataset_path_faces, dataset_path_cartoons, batch_size):

    path_faces = root_path + dataset_path_faces
    path_cartoons = root_path + dataset_path_cartoons

    transform_list_faces = get_transforms_config_face()
    transform_list_cartoons = get_transforms_config_cartoon()

    transform_faces = transforms.Compose(transform_list_faces)

    transform_cartoons = transforms.Compose(transform_list_cartoons)

    dataset_faces = torchvision.datasets.ImageFolder(
        path_faces, transform=transform_faces)
    dataset_cartoons = torchvision.datasets.ImageFolder(
        path_cartoons, transform=transform_cartoons)

    train_dataset_faces, test_dataset_faces = torch.utils.data.random_split(dataset_faces,
                                                                            (int(len(dataset_faces)*0.9), len(dataset_faces) - int(len(dataset_faces)*0.9)))

    train_loader_faces = torch.utils.data.DataLoader(
        train_dataset_faces,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    test_loader_faces = torch.utils.data.DataLoader(
        test_dataset_faces,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    train_dataset_cartoons, test_dataset_cartoons = torch.utils.data.random_split(dataset_cartoons,
                                                                                  (int(len(dataset_cartoons)*0.9), len(dataset_cartoons) - int(len(dataset_cartoons)*0.9)))

    train_loader_cartoons = torch.utils.data.DataLoader(
        train_dataset_cartoons,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    test_loader_cartoons = torch.utils.data.DataLoader(
        test_dataset_cartoons,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4)

    return (train_loader_faces, test_loader_faces, train_loader_cartoons, test_loader_cartoons)


def remove_background_image(model, path_filename, output_path):

    output_file = path_filename.split('/')[-1].split('.')[0] + "_wo_bg.jpg"

    out = model.predict_segmentation(
        inp=path_filename,
        out_fname=output_path + output_file
    )

    img_mask = cv2.imread(output_path + output_file)
    img1 = cv2.imread(path_filename)  # READ BGR

    seg_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    _, bg_mask = cv2.threshold(
        seg_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

    bg = cv2.bitwise_or(img1, bg_mask)

    cv2.imwrite(output_path + output_file, bg)


def remove_background(model, path_test_faces, path_segmented_faces):

    path = path_test_faces + 'data/'
    output_path = path_segmented_faces + 'data/'

    dir_path = os.path.dirname(output_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    for filename in tqdm(os.listdir(path)):

        remove_background_image(model, path + filename, output_path)


def get_test_images(model, batch_size, path_test_faces, path_segmented_faces):

    remove_background(model, path_test_faces, path_segmented_faces)

    path_test_images = path_segmented_faces

    transform_list_faces = get_transforms_config_face()
    transform_list_faces += [transforms.CenterCrop(IMAGE_SIZE)]

    transform = transforms.Compose(transform_list_faces)

    dataset_test_images = torchvision.datasets.ImageFolder(
        path_test_images, transform=transform)

    test_loader_images = torch.utils.data.DataLoader(
        dataset_test_images,
        batch_size=batch_size,
        num_workers=4)

    dataiter = iter(test_loader_images)
    test_images = dataiter.next()

    return test_images


def denorm(img_tensors):
    
    return img_tensors * STATS[1][0] + STATS[0][0]


def test_image(model, device, images_faces):

    e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser = model

    e1.eval()
    e2.eval()
    e_shared.eval()
    d_shared.eval()
    d1.eval()
    d2.eval()
    c_dann.eval()
    discriminator1.eval()
    denoiser.eval()

    with torch.no_grad():
        output = e1(images_faces[0].to(device))
        output = e_shared(output)
        output = d_shared(output)
        output = d2(output)
        output = denoiser(output)

    output = denorm(output)

    return output.cpu()


            


def init_optimizers(model, learning_rate_opDisc, learning_rate_opTotal, learning_rate_denoiser, learning_rate_opCdann):

    e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser = model

    optimizerDisc1 = torch.optim.Adam(
        discriminator1.parameters(), lr=learning_rate_opDisc, betas=(0.5, 0.999))

    #listParameters = list(e1.parameters()) + list(e2.parameters()) + list(e_shared.parameters()) + list(d_shared.parameters()) + list(d1.parameters()) + list(d2.parameters()) + list(c_dann.parameters())
    listParameters = list(e1.parameters()) + list(e2.parameters()) + list(e_shared.parameters()) + \
                    list(d_shared.parameters()) + list(d1.parameters()) + list(d2.parameters())
    optimizerTotal = torch.optim.Adam(
        listParameters, lr=learning_rate_opTotal, betas=(0.5, 0.999))

    optimizerDenoiser = torch.optim.Adam(
        denoiser.parameters(), lr=learning_rate_denoiser)

    optimizerCdann = torch.optim.Adam(
        c_dann.parameters(), lr=learning_rate_opCdann, betas=(0.5, 0.999))

    return (optimizerDenoiser, optimizerDisc1, optimizerTotal, optimizerCdann)


