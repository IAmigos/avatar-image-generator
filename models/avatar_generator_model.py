import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from PIL import Image
from keras_segmentation.pretrained import pspnet_101_voc12
import cv2


from .encoder import *
from .decoder import *
from .discriminator import *
from .denoiser import *
from .cdann import *
from utils import *
from losses import *

import wandb
import os
import sys
from tqdm import tqdm
from itertools import cycle


class Avatar_Generator_Model():
    """
    # Methods
    __init__(dict_model): initializer
    dict_model: layers required to perform face-to-image generation (e1, e_shared, d_shared, d2, denoiser)
    generate(face_image, output_path=None): reutrn cartoon generated from given face image, saves it to output path if given
    load_weights(weights_path): loads weights from given path
    """

    def __init__(self, config, use_wandb=True):
        self.use_wandb = use_wandb
        self.config = config
        self.device = torch.device("cuda:" + (os.getenv('N_CUDA')if os.getenv('N_CUDA') else "0") if self.config.use_gpu and torch.cuda.is_available() else "cpu")
        
        self.segmentation = pspnet_101_voc12()
        self.e1, self.e2, self.d1, self.d2, self.e_shared, self.d_shared, self.c_dann, self.discriminator1, self.denoiser = self.init_model(self.device, self.config.dropout_rate_eshared, self.use_wandb)
        

    def init_model(self, device, dropout_rate_eshared, use_wandb=True):
        
        e1 = Encoder()
        e2 = Encoder()
        e_shared = Eshared(dropout_rate_eshared)
        d_shared = Dshared()
        d1 = Decoder()
        d2 = Decoder()
        # c_dann = Cdann(config.dropout_rate_cdann)
        c_dann = Critic()
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


    def generate(self, path_filename, output_path):
        face = self.__extract_face(path_filename, output_path)
        return self.__to_cartoon(face, output_path)


    def load_weights(self, weights_path):

        self.e1.load_state_dict(torch.load(
            weights_path + 'e1.pth', map_location=torch.device(self.device)))

        self.e_shared.load_state_dict(
            torch.load(weights_path + 'e_shared.pth', map_location=torch.device(self.device)))

        self.e2.load_state_dict(
            torch.load(weights_path + 'e2.pth', map_location=torch.device(self.device)))

        self.d_shared.load_state_dict(
            torch.load(weights_path + 'd_shared.pth', map_location=torch.device(self.device)))

        self.d2.load_state_dict(torch.load(
            weights_path + 'd2.pth', map_location=torch.device(self.device)))

        self.d1.load_state_dict(torch.load(
            weights_path + 'd1.pth', map_location=torch.device(self.device)))

        self.denoiser.load_state_dict(
            torch.load(weights_path + 'denoiser.pth', map_location=torch.device(self.device)))

        self.discriminator1.load_state_dict(
            torch.load(weights_path + 'disc1.pth', map_location=torch.device(self.device)))

        self.c_dann.load_state_dict(
            torch.load(weights_path + 'c_dann.pth', map_location=torch.device(self.device)))


    def __extract_face(self, path_filename, output_path):
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

        transform_list_faces = get_transforms_config_face()
        transform = transforms.Compose(transform_list_faces)
        face = transform(face).float()
        X = face.unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.e1(X)
            output = self.e_shared(output)
            output = self.d_shared(output)
            output = self.d2(output)
            output = self.denoiser(output)
        
        output = denorm(output)
        output = output[0]

        torchvision.utils.save_image(tensor=output, fp=output_path)

        return (torchvision.transforms.ToPILImage()(output), output)


    def train_crit_repeats(self, crit_opt, faces_encoder, cartoons_encoder, crit_repeats=5):

        mean_iteration_critic_loss = torch.zeros(1).to(self.device)
        for _ in range(crit_repeats):
            ### Update critic ###
            crit_opt.zero_grad()
            crit_fake_pred = self.c_dann(faces_encoder)  # bs, 1024
            crit_real_pred = self.c_dann(cartoons_encoder)

            epsilon = torch.rand(len(cartoons_encoder), 1,
                                device=self.device, requires_grad=True)
            gradient = get_gradient(
                self.c_dann, cartoons_encoder, faces_encoder, epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(
                crit_fake_pred, crit_real_pred, gp, 10)

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            crit_opt.step()
        loss_dann = mean_iteration_critic_loss
        
        return loss_dann

    def train_step(self, train_loader_faces, train_loader_cartoons, optimizers, criterion_bc, criterion_l1, criterion_l2):
        
        optimizerDenoiser, optimizerDisc1, optimizerTotal, crit_opt = optimizers

        self.e1.train()
        self.e2.train()
        self.e_shared.train()
        self.d_shared.train()
        self.d1.train()
        self.d2.train()
        self.c_dann.train()
        self.discriminator1.train()
        self.denoiser.train()

        for faces_batch, cartoons_batch in zip(cycle(train_loader_faces), train_loader_cartoons):

            faces_batch, _ = faces_batch
            faces_batch = Variable(faces_batch.type(torch.Tensor))
            #class_faces = Variable(torch.zeros(faces_batch.size(0)))
            faces_batch = faces_batch.to(self.device)
            #class_faces = class_faces.to(device)

            cartoons_batch, _ = cartoons_batch
            cartoons_batch = Variable(cartoons_batch.type(torch.Tensor))
            #class_cartoons = Variable(torch.ones(cartoons_batch.size(0)))
            cartoons_batch = cartoons_batch.to(self.device)
            #class_cartoons = class_cartoons.to(device)

            self.e1.zero_grad()
            self.e2.zero_grad()
            self.e_shared.zero_grad()
            self.d_shared.zero_grad()
            self.d1.zero_grad()
            self.d2.zero_grad()
            self.c_dann.zero_grad()

            if faces_batch.shape != cartoons_batch.shape:
                continue

            # architecture
            faces_enc1 = self.e1(faces_batch)
            faces_encoder = self.e_shared(faces_enc1)
            faces_decoder = self.d_shared(faces_encoder)
            faces_rec = self.d1(faces_decoder)
            cartoons_construct = self.d2(faces_decoder)
            cartoons_construct_enc2 = self.e2(cartoons_construct)
            cartoons_construct_encoder = self.e_shared(cartoons_construct_enc2)

            cartoons_enc2 = self.e2(cartoons_batch)
            cartoons_encoder = self.e_shared(cartoons_enc2)
            cartoons_decoder = self.d_shared(cartoons_encoder)
            cartoons_rec = self.d2(cartoons_decoder)
            faces_construct = self.d1(cartoons_decoder)
            faces_construct_enc1 = self.e1(faces_construct)
            faces_construct_encoder = self.e_shared(faces_construct_enc1)

            #label_output_face = c_dann(faces_encoder)
            #label_output_cartoon = c_dann(cartoons_encoder)

            # train critic(cdann)

            loss_dann = self.train_crit_repeats(crit_opt, faces_encoder, cartoons_encoder, crit_repeats=5)

            # train generator

            loss_rec1 = L2_norm(faces_batch, faces_rec)
            loss_rec2 = L2_norm(cartoons_batch, cartoons_rec)
            loss_rec = loss_rec1 + loss_rec2

            # loss_dann = criterion_bc(label_output_face.squeeze(), torch.zeros_like(label_output_face.squeeze(
            # ), device=device)) + criterion_bc(label_output_cartoon.squeeze(), torch.ones_like(label_output_cartoon.squeeze(), device=device))

            loss_sem1 = L1_norm(faces_encoder.detach(), cartoons_construct_encoder)
            loss_sem2 = L1_norm(cartoons_encoder.detach(), faces_construct_encoder)
            loss_sem = loss_sem1 + loss_sem2

            # teach loss
            #faces_embedding = resnet(faces_batch.squeeze())
            #loss_teach = L1_norm(faces_embedding.squeeze(), faces_encoder)
            # constant until train facenet
            loss_teach = torch.Tensor([1]).requires_grad_()
            loss_teach = loss_teach.to(self.device)

            # class_faces.fill_(1)

            output = self.discriminator1(cartoons_construct)

            loss_gen1 = criterion_bc(output.squeeze(), torch.ones_like(
                output.squeeze(), device=self.device))

            #it has been deleted config.wDann_loss*loss_dann
            loss_total = self.config.wRec_loss*loss_rec  + \
                self.config.wSem_loss*loss_sem + self.config.wGan_loss * \
                loss_gen1 + self.config.wTeach_loss*loss_teach
            loss_total.backward()

            optimizerTotal.step()

            # discriminator face(1)->cartoon(2)
            self.discriminator1.zero_grad()
            # train discriminator with real cartoon images
            output_real = self.discriminator1(cartoons_batch)
            loss_disc1_real_cartoons = self.config.wGan_loss * \
                criterion_bc(output_real.squeeze(), torch.ones_like(
                    output_real.squeeze(), device=self.device))
            # loss_disc1_real_cartoons.backward()

            # train discriminator with fake cartoon images
            # class_faces.fill_(0)
            faces_enc1 = self.e1(faces_batch).detach()
            faces_encoder = self.e_shared(faces_enc1).detach()
            faces_decoder = self.d_shared(faces_encoder).detach()
            cartoons_construct = self.d2(faces_decoder).detach()
            output_fake = self.discriminator1(cartoons_construct)
            loss_disc1_fake_cartoons = self.config.wGan_loss * \
                criterion_bc(output_fake.squeeze(), torch.zeros_like(
                    output_fake.squeeze(), device=self.device))
            # loss_disc1_fake_cartoons.backward()

            loss_disc1 = loss_disc1_real_cartoons + loss_disc1_fake_cartoons
            loss_disc1.backward()
            optimizerDisc1.step()

            # Denoiser
            self.denoiser.zero_grad()
            cartoons_denoised = self.denoiser(cartoons_rec.detach())

            # Train Denoiser

            loss_denoiser = L2_norm(cartoons_batch, cartoons_denoised)
            loss_denoiser.backward()

            optimizerDenoiser.step()

        return loss_rec1, loss_rec2, loss_dann, loss_sem1, loss_sem2, loss_disc1, loss_gen1, loss_total, loss_denoiser, loss_teach, loss_disc1_real_cartoons, loss_disc1_fake_cartoons


    def train(self):    

        if self.config.use_gpu and torch.cuda.is_available():
            print("Training in " + torch.cuda.get_device_name(0))  
        else:
            print("Training in CPU")

        if self.config.save_weights:
            path_save_weights = self.config.root_path + self.config.save_path
        try:
            os.mkdir(path_save_weights)
        except OSError:
            pass

        model = (self.e1, self.e2, self.d1, self.d2, self.e_shared, self.d_shared, self.c_dann, self.discriminator1, self.denoiser)

        train_loader_faces, test_loader_faces, train_loader_cartoons, test_loader_cartoons = get_datasets(self.config.root_path, self.config.dataset_path_faces, self.config.dataset_path_cartoons, self.config.batch_size)
        optimizers = init_optimizers(model, self.config.learning_rate_opDisc, self.config.learning_rate_opTotal, self.config.learning_rate_denoiser)

        train_loss_rec1 = []
        train_loss_rec2 = []
        train_loss_cdan = []
        train_loss_sem1 = []
        train_loss_sem2 = []
        train_disc1_real = []
        train_disc1_fake = []
        train_disc1 = []
        train_gen1 = []
        train_loss_total = []
        train_loss_denoiser = []
        train_loss_teacher = []

        criterion_bc = nn.BCELoss()
        criterion_l1 = nn.L1Loss()
        criterion_l2 = nn.MSELoss()

        criterion_bc.to(self.device)
        criterion_l1.to(self.device)
        criterion_l2.to(self.device)

        images_faces_to_test = get_test_images(self.segmentation, self.config.batch_size, self.config.root_path + self.config.dataset_path_test_faces, self.config.root_path + self.config.dataset_path_segmented_faces)

        for epoch in tqdm(range(self.config.num_epochs)):
            loss_rec1, loss_rec2, loss_dann, loss_sem1, loss_sem2, loss_disc1, loss_gen1, loss_total, loss_denoiser, loss_teach, loss_disc1_real_cartoons, loss_disc1_fake_cartoons = self.train_step(train_loader_faces, train_loader_cartoons, optimizers, criterion_bc, criterion_l1, criterion_l2)
            generated_images = test_image(model, self.device, images_faces_to_test)


            if self.use_wandb:
                wandb.log({"train_epoch": epoch+1,
                        "Generated images": [wandb.Image(img) for img in generated_images],
                        "loss_rec1": loss_rec1.item(),
                        "loss_rec2": loss_rec2.item(),
                        "loss_dann": loss_dann.item(),
                        "loss_semantic12": loss_sem1.item(),
                        "loss_semantic21": loss_sem2.item(),
                        "loss_disc1_real_cartoons": loss_disc1_real_cartoons.item(),
                        "loss_disc1_fake_cartoons": loss_disc1_fake_cartoons.item(),
                        "loss_disc1": loss_disc1.item(),
                        "loss_gen1": loss_gen1.item(),
                        "loss_teach": loss_teach.item(),
                        "loss_total": loss_total.item()})

            if self.config.save_weights and ((epoch+1) % int(self.config.num_epochs/self.config.num_backups)) == 0:
                path_save_epoch = path_save_weights + 'epoch_{}'.format(epoch+1)
                try:
                    os.mkdir(path_save_epoch)
                except OSError:
                    pass
                save_weights(model, path_save_epoch, self.use_wandb)


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
            print('Epoch [{}/{}], Loss rec1: {:.4f}'.format(epoch +
                                                            1, self.config.num_epochs, loss_rec1.item()))
            print('Epoch [{}/{}], Loss rec2: {:.4f}'.format(epoch +
                                                            1, self.config.num_epochs, loss_rec2.item()))
            print('Epoch [{}/{}], Loss dann: {:.4f}'.format(epoch +
                                                            1, self.config.num_epochs, loss_dann.item()))
            print('Epoch [{}/{}], Loss semantic 1->2: {:.4f}'.format(epoch +
                                                                    1, self.config.num_epochs, loss_sem1.item()))
            print('Epoch [{}/{}], Loss semantic 2->1: {:.4f}'.format(epoch +
                                                                    1, self.config.num_epochs, loss_sem2.item()))
            print('Epoch [{}/{}], Loss disc1 real cartoons: {:.4f}'.format(epoch +
                                                                        1, self.config.num_epochs, loss_disc1_real_cartoons.item()))
            print('Epoch [{}/{}], Loss disc1 fake cartoons: {:.4f}'.format(epoch +
                                                                        1, self.config.num_epochs, loss_disc1_fake_cartoons.item()))
            print('Epoch [{}/{}], Loss disc1: {:.4f}'.format(epoch +
                                                            1, self.config.num_epochs, loss_disc1.item()))
            print('Epoch [{}/{}], Loss gen1: {:.4f}'.format(epoch +
                                                            1, self.config.num_epochs, loss_gen1.item()))
            print('Epoch [{}/{}], Loss teach: {:.4f}'.format(epoch +
                                                            1, self.config.num_epochs, loss_teach.item()))
            print('Epoch [{}/{}], Loss total: {:.4f}'.format(epoch +
                                                            1, self.config.num_epochs, loss_total.item()))
            print('Epoch [{}/{}], Loss denoiser: {:.4f}'.format(epoch +
                                                                1, self.config.num_epochs, loss_denoiser.item()))

        if self.use_wandb:
            wandb.finish()
