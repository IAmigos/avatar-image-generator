# wandb login 17d2772d85cbda79162bd975e45fdfbf3bb18911

import argparse
from utils import *
from losses import *
from models import *

#from facenet_pytorch import InceptionResnetV1
import wandb
import math
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
import os, sys

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

torch.manual_seed(0)
np.random.seed(0)


def train(config, model, device, train_loader_faces, train_loader_cartoons, optimizers, criterion_bc, criterion_l1, criterion_l2):
  
  e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser = model
  optimizerDenoiser, optimizerDisc1, optimizerTotal = optimizers 
  
  e1.train()
  e2.train()
  e_shared.train()
  d_shared.train()
  d1.train()
  d2.train()
  c_dann.train()
  discriminator1.train()
  denoiser.train()

  for faces_batch, cartoons_batch in zip(train_loader_faces, train_loader_cartoons):

    faces_batch,_ = faces_batch
    faces_batch = Variable(faces_batch.type(torch.Tensor))
    #class_faces = Variable(torch.zeros(faces_batch.size(0)))
    faces_batch = faces_batch.to(device)
    #class_faces = class_faces.to(device)
    
    cartoons_batch,_ = cartoons_batch
    cartoons_batch = Variable(cartoons_batch.type(torch.Tensor))
    #class_cartoons = Variable(torch.ones(cartoons_batch.size(0)))
    cartoons_batch = cartoons_batch.to(device)
    #class_cartoons = class_cartoons.to(device)
    
    e1.zero_grad()
    e2.zero_grad()
    e_shared.zero_grad()
    d_shared.zero_grad()
    d1.zero_grad()
    d2.zero_grad()
    c_dann.zero_grad()

    #architecture
    faces_enc1 = e1(faces_batch)
    faces_encoder = e_shared(faces_enc1)
    faces_decoder = d_shared(faces_encoder)
    faces_rec = d1(faces_decoder)
    cartoons_construct = d2(faces_decoder)
    cartoons_construct_enc2 = e2(cartoons_construct)
    cartoons_construct_encoder = e_shared(cartoons_construct_enc2)

    cartoons_enc2 = e2(cartoons_batch)
    cartoons_encoder = e_shared(cartoons_enc2)
    cartoons_decoder = d_shared(cartoons_encoder)
    cartoons_rec = d2(cartoons_decoder)
    faces_construct = d1(cartoons_decoder)
    faces_construct_enc1 = e1(faces_construct)
    faces_construct_encoder = e_shared(faces_construct_enc1)

    label_output_face = c_dann(faces_encoder)
    label_output_cartoon = c_dann(cartoons_encoder)

    #train generator

    loss_rec1 = L2_norm(faces_batch, faces_rec)
    loss_rec2 = L2_norm(cartoons_batch, cartoons_rec)
    loss_rec =  loss_rec1 + loss_rec2

    loss_dann = criterion_bc(label_output_face.squeeze(), torch.zeros_like(label_output_face.squeeze(), device=device)) +  criterion_bc(label_output_cartoon.squeeze(), torch.ones_like(label_output_cartoon.squeeze(), device=device))

    loss_sem1 = L1_norm(faces_encoder.detach(), cartoons_construct_encoder) 
    loss_sem2 = L1_norm(cartoons_encoder.detach(), faces_construct_encoder) 
    loss_sem = loss_sem1 + loss_sem2

    #teach loss
    #faces_embedding = resnet(faces_batch.squeeze())
    #loss_teach = L1_norm(faces_embedding.squeeze(), faces_encoder)
    #constant until train facenet
    loss_teach = torch.Tensor([1]).requires_grad_() 
    loss_teach = loss_teach.to(device)

    #class_faces.fill_(1)

    output = discriminator1(cartoons_construct)

    loss_gen1 = criterion_bc(output.squeeze(), torch.ones_like(output.squeeze(), device=device))



    loss_total = config.wRec_loss*loss_rec + config.wDann_loss*loss_dann + config.wSem_loss*loss_sem + config.wGan_loss*loss_gen1 + config.wTeach_loss*loss_teach
    loss_total.backward()


    optimizerTotal.step()



    #discriminator face(1)->cartoon(2)
    discriminator1.zero_grad()
      #train discriminator with real cartoon images
    output_real = discriminator1(cartoons_batch)
    loss_disc1_real_cartoons = config.wGan_loss * criterion_bc(output_real.squeeze(), torch.ones_like(output_real.squeeze(), device=device))
    #loss_disc1_real_cartoons.backward()

      #train discriminator with fake cartoon images
    #class_faces.fill_(0)
    faces_enc1 = e1(faces_batch).detach()
    faces_encoder = e_shared(faces_enc1).detach()
    faces_decoder = d_shared(faces_encoder).detach()
    cartoons_construct = d2(faces_decoder).detach()  
    output_fake = discriminator1(cartoons_construct)
    loss_disc1_fake_cartoons = config.wGan_loss * criterion_bc(output_fake.squeeze(), torch.zeros_like(output_fake.squeeze(), device=device))
    #loss_disc1_fake_cartoons.backward()

    loss_disc1 = loss_disc1_real_cartoons + loss_disc1_fake_cartoons
    loss_disc1.backward()
    optimizerDisc1.step()

    # Denoiser
    denoiser.zero_grad()
    cartoons_denoised = denoiser(cartoons_rec.detach())

    # Train Denoiser

    loss_denoiser = L2_norm(cartoons_batch, cartoons_denoised)
    loss_denoiser.backward()

    optimizerDenoiser.step()


  return loss_rec1, loss_rec2, loss_dann,loss_sem1, loss_sem2, loss_disc1, loss_gen1, loss_total, loss_denoiser, loss_teach, loss_disc1_real_cartoons, loss_disc1_fake_cartoons



def model_train(config_file, use_wandb=True):

  if use_wandb:
    wandb.init(project="avatar_image_generator")
    wandb.watch_called = False


  config = configure_model(config_file, use_wandb)

  device = torch.device("cuda:0" if config.use_gpu and torch.cuda.is_available() else "cpu")
  #device = cf.DEVICE
  #device = torch.device(device)
  if config.use_gpu and torch.cuda.is_available():
    print("Training in " + torch.cuda.get_device_name(0))  
  else:
    print("Training in CPU")

  if config.save_weights:
    path_save_weights = config.root_path + config.save_path
    try:
        os.mkdir(path_save_weights)
    except OSError:
        pass

  #resnet = InceptionResnetV1(pretrained='vggface2').eval()
  #resnet.to(device)

  logging = init_logger(log_file='logfile.log',log_dir=path_save_weights)

  train_loader_faces, test_loader_faces, train_loader_cartoons, test_loader_cartoons = get_datasets(config)

  model = init_model(device, config, use_wandb)
  optimizers = init_optimizers(model, config)

  train_loss_rec1 = []
  train_loss_rec2 = []
  train_loss_cdan = []
  train_loss_sem1 = []
  train_loss_sem2 = []
  train_disc1_real =[]
  train_disc1_fake =[]
  train_disc1 = []
  train_gen1 = []
  train_loss_total = []
  train_loss_denoiser = []
  train_loss_teacher = []

  criterion_bc = nn.BCELoss()
  criterion_l1 = nn.L1Loss()
  criterion_l2 = nn.MSELoss()
  #criterion_l1 = L1_norm()
  #criterion_l2 = L2_norm()

  criterion_bc.to(device) 
  criterion_l1.to(device)
  criterion_l2.to(device)
  

  images_faces_to_test = get_test_images(config, config.root_path + config.dataset_path_test_faces, config.root_path + config.dataset_path_segmented_faces)

  for epoch in tqdm(range(config.num_epochs)):
    loss_rec1, loss_rec2, loss_dann,loss_sem1, loss_sem2, loss_disc1, loss_gen1, loss_total, loss_denoiser, loss_teach, loss_disc1_real_cartoons, loss_disc1_fake_cartoons = train(config, model, device, train_loader_faces, train_loader_cartoons, optimizers, criterion_bc, criterion_l1, criterion_l2)
    generated_images = test_image(model, device, images_faces_to_test)


    logging.info('Train Epoch [{}/{}], Loss rec1: {:.4f}, Loss rec2: {:.4f},'
                                      ' Loss dann: {:.4f}, Loss semantic 1->2: {:.4f}, Loss semantic 2->1: {:.4f},'
                                      ' Loss disc1: {:.4f}, Loss gen1: {:.4f}, Loss teach: {:.4f}, Loss total: {:.4f}'
                                      ' Loss disc1 real cartoons: {:.4f}, Loss disc1 fake cartoons: {:.4f}'
                                      .format(epoch+1, config.num_epochs, loss_rec1.item(),
                                              loss_rec2.item(), loss_dann.item(),
                                              loss_sem1.item(), loss_sem2.item(),
                                              loss_disc1.item(), loss_gen1.item(), loss_teach.item(),
                                              loss_total.item(), loss_disc1_real_cartoons.item(), loss_disc1_fake_cartoons.item()))
    if use_wandb:
      wandb.log({"train_epoch":epoch+1,
                "Generated images": [wandb.Image(img) for img in generated_images],
                "loss_rec1":loss_rec1.item(),
                "loss_rec2":loss_rec2.item(),
                "loss_dann":loss_dann.item(),
                "loss_semantic12":loss_sem1.item(),
                "loss_semantic21":loss_sem2.item(),
                "loss_disc1_real_cartoons":loss_disc1_real_cartoons.item(),
                "loss_disc1_fake_cartoons":loss_disc1_fake_cartoons.item(),
                "loss_disc1":loss_disc1.item(),
                "loss_gen1":loss_gen1.item(),
                "loss_teach":loss_teach.item(), 
                "loss_total":loss_total.item()})


    if config.save_weights and ((epoch+1)% int(config.num_epochs/config.num_backups))==0:
      path_save_epoch = path_save_weights + 'epoch_{}'.format(epoch+1)
      try:
          os.mkdir(path_save_epoch)
      except OSError:
          pass
      save_weights(model, path_save_epoch, use_wandb)      
      logging.info(f'Checkpoint {epoch + 1} saved !')

    train_loss_rec1.append(loss_rec1.item())
    train_loss_rec2.append(loss_rec2.item())
    train_loss_cdan.append(loss_dann.item())
    train_loss_sem1.append(loss_sem1.item())
    train_loss_sem2.append(loss_sem2.item())
    train_disc1.append(loss_disc1.item())
    train_gen1.append(loss_gen1.item())
    train_loss_total.append(loss_total.item())
    train_loss_denoiser.append(loss_denoiser.item())
    train_loss_teacher.append(loss_teach.item())
    train_disc1_real.append(loss_disc1_real_cartoons)
    train_disc1_fake.append(loss_disc1_fake_cartoons)

    print("Losses")
    print('Epoch [{}/{}], Loss rec1: {:.4f}'.format(epoch+1, config.num_epochs, loss_rec1.item()))
    print('Epoch [{}/{}], Loss rec2: {:.4f}'.format(epoch+1, config.num_epochs, loss_rec2.item()))
    print('Epoch [{}/{}], Loss dann: {:.4f}'.format(epoch+1, config.num_epochs, loss_dann.item()))
    print('Epoch [{}/{}], Loss semantic 1->2: {:.4f}'.format(epoch+1, config.num_epochs, loss_sem1.item()))
    print('Epoch [{}/{}], Loss semantic 2->1: {:.4f}'.format(epoch+1, config.num_epochs, loss_sem2.item()))    
    print('Epoch [{}/{}], Loss disc1 real cartoons: {:.4f}'.format(epoch+1, config.num_epochs, loss_disc1_real_cartoons.item()))
    print('Epoch [{}/{}], Loss disc1 fake cartoons: {:.4f}'.format(epoch+1, config.num_epochs, loss_disc1_fake_cartoons.item()))
    print('Epoch [{}/{}], Loss disc1: {:.4f}'.format(epoch+1, config.num_epochs, loss_disc1.item()))
    print('Epoch [{}/{}], Loss gen1: {:.4f}'.format(epoch+1, config.num_epochs, loss_gen1.item()))
    print('Epoch [{}/{}], Loss teach: {:.4f}'.format(epoch+1, config.num_epochs, loss_teach.item()))
    print('Epoch [{}/{}], Loss total: {:.4f}'.format(epoch+1, config.num_epochs, loss_total.item()))
    print('Epoch [{}/{}], Loss denoiser: {:.4f}'.format(epoch+1, config.num_epochs, loss_denoiser.item()))

  if use_wandb:
    wandb.finish()

if __name__=='__main__':
  args = parse_arguments()
  use_wandb = args.wandb

  model_train('config.json', use_wandb)
