import wandb
import math
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys
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

from keras_segmentation.pretrained import pspnet_50_ADE_20K , pspnet_101_cityscapes, pspnet_101_voc12
import cv2
import helper
import json
from models import *

#from facenet_pytorch import InceptionResnetV1

def parse_arguments():
  ap = argparse.ArgumentParser()
  ap.add_argument('-w','--wandb', default=False, action='store_true',
   help="use weights and biases")
  ap.add_argument('-nw  ','--no-wandb', dest='wandb', action='store_false',
   help="not use weights and biases")

  args = ap.parse_args()

  return args

def parse_configuration(config_file):
    """Loads config file if a string was passed
        and returns the input if a dictionary was passed.
    """
    if isinstance(config_file, str):
        with open(config_file,'r') as json_file:
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



def configure_model(config_file, use_wandb=False):

  config_file = parse_configuration(config_file)

  if use_wandb:
    config = wandb.config                                    
  else:
    config = type("configuration", (object,), {})

  config.model_path = config_file["server_config"]["model_path"]
  config.device = config_file["server_config"]["device"]
  config.download_directory = config_file["server_config"]["download_directory"]

  config.root_path = config_file["train_dataset_params"]["root_path"]
  config.dataset_path_faces = config_file["train_dataset_params"]["dataset_path_faces"]
  config.dataset_path_cartoons = config_file["train_dataset_params"]["dataset_path_cartoons"]
  config.dataset_path_test_faces = config_file["train_dataset_params"]["dataset_path_test_faces"]
  config.dataset_path_segmented_faces = config_file["train_dataset_params"]["dataset_path_segmented_faces"]
  config.dataset_path_output_faces = config_file["train_dataset_params"]["dataset_path_output_faces"]
  config.batch_size = config_file["train_dataset_params"]["loader_params"]["batch_size"]
  config.image_size = config_file["train_dataset_params"]["loader_params"]["image_size"]
  config.shuffle = config_file["train_dataset_params"]["loader_params"]["shuffle"]
  config.workers = config_file["train_dataset_params"]["loader_params"]["workers"]
  config.seed = config_file["train_dataset_params"]["loader_params"]["seed"]

  config.save_weights = config_file["train_dataset_params"]["save_weights"]
  config.num_backups = config_file["train_dataset_params"]["num_backups"]
  config.save_path = config_file["train_dataset_params"]["save_path"]

  config.dropout_rate_eshared = config_file["model_hparams"]["dropout_rate_eshared"]
  config.dropout_rate_cdann = config_file["model_hparams"]["dropout_rate_cdann"]
  config.is_train = config_file["model_hparams"]["is_train"]
  config.num_epochs = config_file["model_hparams"]["num_epochs"]
  config.learning_rate_opTotal = config_file["model_hparams"]["learning_rate_opTotal"]
  config.learning_rate_opDisc = config_file["model_hparams"]["learning_rate_opDisc"]
  config.b1_disc = config_file["model_hparams"]["b1_disc"]
  config.learning_rate_opCdann = config_file["model_hparams"]["learning_rate_opCdann"]
  config.b1_cdann = config_file["model_hparams"]["b1_cdann"]
  config.learning_rate_denoiser = config_file["model_hparams"]["learning_rate_denoiser"]
  config.wRec_loss = config_file["model_hparams"]["wRec_loss"]
  config.wDann_loss = config_file["model_hparams"]["wDann_loss"]
  config.wSem_loss = config_file["model_hparams"]["wSem_loss"]
  config.wGan_loss = config_file["model_hparams"]["wGan_loss"]
  config.wTeach_loss = config_file["model_hparams"]["wTeach_loss"]
  config.use_gpu = config_file["model_hparams"]["use_gpu"]

  return config


def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    nn.init.kaiming_uniform_(m.weight.data )

  elif classname.find('BatchNorm') != -1:
    nn.init.normal_(m.weight.data, 1.0, 0.02)
    nn.init.constant_(m.bias.data, 0)


def save_weights(model, path_gen, path_sub, use_wandb=True):
  e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser = model

  torch.save(e1.state_dict(), os.path.join(path_sub, 'e1.pth'))
  torch.save(e2.state_dict(), os.path.join(path_sub, 'e2.pth'))
  torch.save(e_shared.state_dict(), os.path.join(path_sub, 'e_shared.pth'))
  torch.save(d_shared.state_dict(), os.path.join(path_sub, 'd_shared.pth'))
  torch.save(d1.state_dict(), os.path.join(path_sub, 'd1.pth'))
  torch.save(d2.state_dict(), os.path.join(path_sub, 'd2.pth'))
  torch.save(c_dann.state_dict(), os.path.join(path_sub, 'c_dann.pth'))
  torch.save(discriminator1.state_dict(), os.path.join(path_sub, 'disc1.pth'))
  torch.save(denoiser.state_dict(), os.path.join(path_sub, 'denoiser.pth'))

  if use_wandb:
    wandb.save(os.path.join(path_sub,'*.pth'),base_path='/'.join(path_gen.split('/')[:-2]))



def get_datasets(config):

  path_faces = config.root_path + config.dataset_path_faces
  path_cartoons = config.root_path + config.dataset_path_cartoons

  transform_faces = transforms.Compose([
                transforms.Resize((config.image_size,config.image_size)) ,
                transforms.ToTensor()
                ])

  transform_cartoons = transforms.Compose([
                transforms.CenterCrop(400),
                transforms.Resize((config.image_size,config.image_size)) ,
                transforms.ToTensor()
                ])

  dataset_faces = torchvision.datasets.ImageFolder(path_faces, transform=transform_faces)
  dataset_cartoons = torchvision.datasets.ImageFolder(path_cartoons, transform=transform_cartoons)

  train_dataset_faces, test_dataset_faces = torch.utils.data.random_split(dataset_faces,
                                                                                (int(len(dataset_faces)*0.9),len(dataset_faces) - int(len(dataset_faces)*0.9)))


  train_loader_faces = torch.utils.data.DataLoader(
      train_dataset_faces,
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      num_workers = config.workers)

  test_loader_faces = torch.utils.data.DataLoader(
      test_dataset_faces,
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      num_workers = config.workers)


  train_dataset_cartoons, test_dataset_cartoons = torch.utils.data.random_split(dataset_cartoons,
                                                                                  (int(len(dataset_cartoons)*0.9),len(dataset_cartoons) - int(len(dataset_cartoons)*0.9)))


  train_loader_cartoons = torch.utils.data.DataLoader(
      train_dataset_cartoons,
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      num_workers = config.workers)

  test_loader_cartoons = torch.utils.data.DataLoader(
      test_dataset_cartoons,
      batch_size=config.batch_size,
      shuffle=config.shuffle,
      num_workers = config.workers)

  return (train_loader_faces, test_loader_faces, train_loader_cartoons, test_loader_cartoons)


def remove_background_image(model, path_filename, output_path):


    output_file = path_filename.split('/')[-1].split('.')[0] + "_wo_bg.jpg"

    out = model.predict_segmentation(
      inp= path_filename,
      out_fname= output_path + output_file
    )

    img_mask = cv2.imread(output_path + output_file)
    img1 = cv2.imread(path_filename) #READ BGR

    seg_gray = cv2.cvtColor(img_mask, cv2.COLOR_BGR2GRAY)
    _,bg_mask = cv2.threshold(seg_gray, 0, 255, cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)

    bg_mask = cv2.cvtColor(bg_mask, cv2.COLOR_GRAY2BGR)

    bg = cv2.bitwise_or(img1, bg_mask)
    
    cv2.imwrite(output_path + output_file, bg)



def remove_background(path_test_faces, path_segmented_faces):
  model = pspnet_101_voc12()

  path = path_test_faces + 'data/'
  output_path = path_segmented_faces + 'data/'

  dir_path = os.path.dirname(output_path)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)

  for filename in tqdm(os.listdir(path)):

    remove_background_image(model, path + filename, output_path)


def get_test_images(config, path_test_faces, path_segmented_faces):

  remove_background(path_test_faces, path_segmented_faces)

  path_test_images = path_segmented_faces

  transform = transforms.Compose([
                                
                  transforms.Resize((config.image_size,config.image_size)) ,
                  transforms.CenterCrop(64), 
                  transforms.ToTensor(),  
                  ])

  dataset_test_images = torchvision.datasets.ImageFolder(path_test_images, transform=transform)

  test_loader_images = torch.utils.data.DataLoader(
      dataset_test_images,
      batch_size=config.batch_size,
      num_workers = config.workers)

  dataiter = iter(test_loader_images)
  test_images = dataiter.next()


  return test_images


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
    
  return output.cpu()


def init_optimizers(model, config):

  e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser = model

  listDisc1 = list(discriminator1.parameters())
  optimizerDisc1 = torch.optim.Adam(listDisc1, lr=config.learning_rate_opDisc, betas=(config.b1_disc, 0.999))

  listParameters = list(e1.parameters()) + list(e2.parameters()) + list(e_shared.parameters()) + list(d_shared.parameters()) + list(d1.parameters()) + list(d2.parameters())

  optimizerTotal = torch.optim.Adam(listParameters, lr=config.learning_rate_opTotal)

  optimizerCdann = torch.optim.Adam(c_dann.parameters(), lr=config.learning_rate_opCdann, betas=(config.b1_cdann, 0.999))

  optimizerDenoiser = torch.optim.Adam(denoiser.parameters(), lr=config.learning_rate_denoiser)

  return (optimizerDenoiser, optimizerDisc1, optimizerTotal, optimizerCdann) 


def init_model(device, config, use_wandb=True):

  e1 = Encoder()
  e2 = Encoder()
  e_shared = Eshared(config.dropout_rate_eshared)
  d_shared = Dshared()
  d1 = Decoder()
  d2 = Decoder()
  c_dann = Cdann(config.dropout_rate_cdann)
  discriminator1 = Discriminator()
  denoiser = Denoiser()

  e1.to(device)
  e2.to(device)
  e_shared.to(device)
  d_shared.to(device)
  d1.to(device)
  d2.to(device)
  c_dann.to(device)
  discriminator1.to(device)
  denoiser = denoiser.to(device)
  
  if use_wandb:
    wandb.watch(e1, log="all")
    wandb.watch(e2, log="all")
    wandb.watch(e_shared, log="all")
    wandb.watch(d_shared, log="all")
    wandb.watch(d1, log="all")
    wandb.watch(d2, log="all")
    wandb.watch(c_dann, log="all")
    wandb.watch(discriminator1, log="all")
    wandb.watch(denoiser, log="all")

  return (e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser)





def load_weights_xgan(path_load_weights, e1, e2, e_shared, d_shared, d1, d2, denoiser):
  
  e1.load_state_dict(torch.load(path_load_weights + 'e1.pth'))
  e2.load_state_dict(torch.load(path_load_weights + 'e2.pth'))
  e_shared.load_state_dict(torch.load(path_load_weights + 'e_shared.pth'))
  d_shared.load_state_dict(torch.load(path_load_weights + 'd_shared.pth'))
  d1.load_state_dict(torch.load(path_load_weights + 'd1.pth'))
  d2.load_state_dict(torch.load(path_load_weights + 'd2.pth'))
  denoiser.load_state_dict(torch.load(path_load_weights + 'denoiser.pth'))

  return
