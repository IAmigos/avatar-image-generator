import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision
from torch.autograd import Variable
from PIL import Image
from keras_segmentation.pretrained import pspnet_101_voc12
import cv2
import numpy as np

from .encoder import *
from .decoder import *
from .discriminator import *
from .denoiser import *
from .cdann import *
from .inception import *
from utils import *
from losses import *
from evaluation import tsne_evaluation

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
        self.mmd_kernel_type = "multiscale"
        self.segmentation = pspnet_101_voc12()
        self.e1, self.e2, self.d1, self.d2, self.e_shared, self.d_shared, self.c_dann, self.discriminator1, self.denoiser, self.inception = self.init_model(self.device, 
                                                                                                                                                            self.config.dropout_rate_eshared,
                                                                                                                                                            self.config.use_critic_dann,
                                                                                                                                                            self.config.use_critic_disc,
                                                                                                                                                            self.config.use_spectral_norm, 
                                                                                                                                                            self.use_wandb)
        

    def init_model(self, device, 
                    dropout_rate_eshared,
                    use_critic_dann, use_critic_disc, 
                    use_spectral_norm, use_wandb=True):
        
        e1 = Encoder()
        e2 = Encoder()
        e_shared = Eshared(dropout_rate_eshared)
        d_shared = Dshared()
        d1 = Decoder()
        d2 = Decoder()
        c_dann = Cdann(use_critic_dann=use_critic_dann, use_spectral_norm=use_spectral_norm)
        discriminator1 = Discriminator(use_critic_disc=use_critic_disc, use_spectral_norm=use_spectral_norm)
        denoiser = Denoiser()
        inception = Inception([Inception.BLOCK_INDEX_BY_DIM[2048]]) #fid
        
        e1.to(device)
        e2.to(device)
        e_shared.to(device)
        d_shared.to(device)
        d1.to(device)
        d2.to(device)
        c_dann.to(device)
        discriminator1.to(device)
        denoiser = denoiser.to(device)
        inception = inception.to(device)

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
            #wandb.watch(inception, log="all")

        return (e1, e2, d1, d2, e_shared, d_shared, c_dann, discriminator1, denoiser, inception)


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

    def get_loss_test_set(self, test_loader_faces, test_loader_cartoons, criterion_bc):
        
        self.e1.eval()
        self.e2.eval()
        self.e_shared.eval()
        self.d_shared.eval()
        self.d1.eval()
        self.d2.eval()
        self.c_dann.eval()
        self.discriminator1.eval()
        self.denoiser.eval()
        self.inception.eval()

        loss_test = []
        cartoons_batch_test = []
        cartoons_construct_test = []

        
        faces_encoder_test = []
        cartoons_encoder_test = []
        cartoons_construct_encoder_test = []
        
        with torch.no_grad():
            for faces_batch, cartoons_batch in zip(cycle(test_loader_faces), test_loader_cartoons):
                
                faces_batch, _ = faces_batch
                faces_batch = Variable(faces_batch.type(torch.Tensor))
                faces_batch = faces_batch.to(self.device)

                cartoons_batch, _ = cartoons_batch
                cartoons_batch = Variable(cartoons_batch.type(torch.Tensor))
                cartoons_batch = cartoons_batch.to(self.device)

                if faces_batch.shape != cartoons_batch.shape:
                    continue

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

                loss_rec1 = L2_norm(faces_batch, faces_rec)
                loss_rec2 = L2_norm(cartoons_batch, cartoons_rec)
                loss_rec = loss_rec1 + loss_rec2

                loss_sem1 = L1_norm(faces_encoder.detach(), cartoons_construct_encoder)
                loss_sem2 = L1_norm(cartoons_encoder.detach(), faces_construct_encoder)
                loss_sem = loss_sem1 + loss_sem2

                loss_teach = torch.Tensor([1])
                loss_teach = loss_teach.to(self.device)

                output = self.discriminator1(cartoons_construct)

                loss_gen1 = criterion_bc(output.squeeze(), torch.ones_like(
                    output.squeeze(), device=self.device))

                loss_total = self.config.wRec_loss*loss_rec  + \
                    self.config.wSem_loss*loss_sem + self.config.wGan_loss * \
                    loss_gen1 + self.config.wTeach_loss*loss_teach

                loss_test.append(loss_total.item())
                
                cartoons_batch_test.append(cartoons_batch)
                cartoons_construct_test.append(cartoons_construct)
                
                faces_encoder_test.append(faces_encoder)
                cartoons_encoder_test.append(cartoons_encoder)
                cartoons_construct_encoder_test.append(cartoons_construct_encoder)
            

#         return np.mean(loss_test)

        cartoons_batch_test = torch.cat(cartoons_batch_test)
        cartoons_construct_test = torch.cat(cartoons_construct_test)
        cartoons_batch_feature_view = cartoons_batch_test.view(cartoons_batch_test.size()[0], -1)
        cartoons_construct_feature_view = cartoons_construct_test.view(cartoons_construct_test.size()[0], -1)
        
        fid_test = fid(cartoons_batch, cartoons_construct, self.inception, self.device)
        mmd_test = MMD(cartoons_batch_feature_view, cartoons_construct_feature_view, self.mmd_kernel_type, self.device)

        # tsne analysis
        faces_encoder_test = torch.cat(faces_encoder_test).cpu()
        cartoons_encoder_test = torch.cat(cartoons_encoder_test).cpu()
        cartoons_construct_encoder_test = torch.cat(cartoons_construct_encoder_test).cpu()
        
        # tsne of faces encoder and cartoons encoder      
        tsne_results_norm, df_feature_vector_info, wandb_scatter_plot_1_fe_ce, img_scatter_plot_1_fe_ce = tsne_evaluation([faces_encoder_test, cartoons_encoder_test], ['faces encoder', 'cartoons encoder'], pca_components=None, perplexity=30, n_iter=1000, save_image=False, save_wandb=self.use_wandb, plot_title='t-SNE evaluation - FE and CE')
        
        # tsne of faces encoder and cartoons construct encoder 
        tsne_results_norm, df_feature_vector_info, wandb_scatter_plot_2_fe_cce, img_scatter_plot_2_fe_cce = tsne_evaluation([faces_encoder_test, cartoons_construct_encoder_test], ['faces encoder', 'cartoons encoder'], pca_components=None, perplexity=30, n_iter=1000, save_image=False, save_wandb=self.use_wandb, plot_title='t-SNE evaluation - FE and CCE')
        
        return np.mean(loss_test), fid_test, mmd_test, wandb_scatter_plot_1_fe_ce, wandb_scatter_plot_2_fe_cce, img_scatter_plot_1_fe_ce, img_scatter_plot_2_fe_cce
        


    def train_crit_repeats(self, opt, fake, real, model, type_model, crit_repeats=5):

        if type_model=="discriminator":
            fake = fake.detach()
            loss_weight =  self.config.wGan_loss
        elif type_model=="cdann":
            loss_weight =  self.config.wDann_loss


        mean_iteration_critic_loss = torch.zeros(1).to(self.device)
        for i in range(crit_repeats):
            ### Update critic ###
            opt.zero_grad()
            lim_inf = i * int(len(fake)/crit_repeats)
            lim_sup = lim_inf + int(len(fake)/crit_repeats) if i < crit_repeats - 1 else 1000
            fake_sample = fake[lim_inf: lim_sup]
            real_sample = real[lim_inf: lim_sup]

            crit_fake_pred = model(fake_sample) 
            crit_real_pred = model(real_sample)

            if type_model=="discriminator":
                epsilon = torch.rand(len(real_sample), 1, 1, 1,
                                    device=self.device, requires_grad=True)
            elif type_model=="cdann":
                epsilon = torch.rand(len(real_sample), 1,
                                    device=self.device, requires_grad=True)
            #print(f"{fake_sample.shape}")
            #print(f"{real_sample.shape}")
            #print(f"epsilon {epsilon.shape}")

            gradient = get_gradient(
                model, real_sample, fake_sample, epsilon)
            gp = gradient_penalty(gradient)
            crit_loss = get_crit_loss(
                crit_fake_pred.squeeze(), crit_real_pred.squeeze(), gp, 10) * self.config.wDann_loss

            # Keep track of the average critic loss in this batch
            mean_iteration_critic_loss += crit_loss / crit_repeats
            # Update gradients
            crit_loss.backward(retain_graph=True)
            # Update optimizer
            opt.step()
        loss = mean_iteration_critic_loss
        
        return loss

    def train_step(self, train_loader_faces, train_loader_cartoons, optimizers, criterion_bc):
        
        optimizerDenoiser, optimizerDisc1, optimizerTotal, optimizerCdann = optimizers

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
            faces_batch = faces_batch.to(self.device)

            cartoons_batch, _ = cartoons_batch
            cartoons_batch = Variable(cartoons_batch.type(torch.Tensor))
            cartoons_batch = cartoons_batch.to(self.device)

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

            # train generator

            #training cdann
            if not self.config.use_critic_dann:
                label_output_face = self.c_dann(faces_encoder)
                label_output_cartoon = self.c_dann(cartoons_encoder)
                loss_dann = criterion_bc(label_output_face.squeeze(), torch.zeros_like(label_output_face.squeeze(
                                        ), device=self.device)) + criterion_bc(label_output_cartoon.squeeze(), torch.ones_like(label_output_cartoon.squeeze(), device=self.device))
                loss_dann.backward(retain_graph=True)
                optimizerCdann.step()
            else:
                # train critic(cdann)
                loss_dann = self.train_crit_repeats(optimizerCdann, faces_encoder, 
                                                    cartoons_encoder, self.c_dann, 
                                                    "cdann", crit_repeats=5)

            loss_rec1 = L2_norm(faces_batch, faces_rec)
            loss_rec2 = L2_norm(cartoons_batch, cartoons_rec)
            loss_rec = loss_rec1 + loss_rec2

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
            if not self.config.use_critic_disc:
                loss_gen1 = criterion_bc(output.squeeze(), torch.ones_like(
                    output.squeeze(), device=self.device))
            else:
                loss_gen1 = get_gen_loss(output.squeeze())


            #it has been deleted config.wDann_loss*loss_dann
            loss_total = self.config.wRec_loss*loss_rec  + \
                self.config.wSem_loss*loss_sem + self.config.wGan_loss * \
                loss_gen1 + self.config.wTeach_loss*loss_teach
            loss_total.backward()
            loss_total += loss_dann

            optimizerTotal.step()

            # discriminator face(1)->cartoon(2)
            self.discriminator1.zero_grad()

            faces_enc1 = self.e1(faces_batch).detach()
            faces_encoder = self.e_shared(faces_enc1).detach()
            faces_decoder = self.d_shared(faces_encoder).detach()
            cartoons_construct = self.d2(faces_decoder).detach()
        
            if not self.config.use_critic_disc:
                # train discriminator with real cartoon images
                output_real = self.discriminator1(cartoons_batch)
                loss_disc1_real_cartoons = self.config.wGan_loss * \
                    criterion_bc(output_real.squeeze(), torch.ones_like(
                        output_real.squeeze(), device=self.device))
                # loss_disc1_real_cartoons.backward()

                # train discriminator with fake cartoon images
                # class_faces.fill_(0)
                
                output_fake = self.discriminator1(cartoons_construct)
                loss_disc1_fake_cartoons = self.config.wGan_loss * \
                    criterion_bc(output_fake.squeeze(), torch.zeros_like(
                        output_fake.squeeze(), device=self.device))
                # loss_disc1_fake_cartoons.backward()

                loss_disc1 = loss_disc1_real_cartoons + loss_disc1_fake_cartoons
                loss_disc1.backward()
                optimizerDisc1.step()
            else:
                loss_disc1 = self.train_crit_repeats(optimizerDisc1, cartoons_construct, 
                                                    cartoons_batch, self.discriminator1, 
                                                    "discriminator", crit_repeats=5)


            # Denoiser
            if self.config.use_denoiser:
                self.denoiser.zero_grad()
                cartoons_denoised = self.denoiser(cartoons_rec.detach())

                # Train Denoiser
                loss_denoiser = L2_norm(cartoons_batch, cartoons_denoised)
                loss_denoiser.backward()

                optimizerDenoiser.step()
            else:
                loss_denoiser = torch.Tensor([0]).requires_grad_()
            
            break #Delete break
            

        return loss_rec1, loss_rec2, loss_dann, loss_sem1, loss_sem2, loss_disc1, loss_gen1, loss_total, loss_denoiser, loss_teach


    def train(self):    

        if self.config.use_gpu and torch.cuda.is_available():
            print("Training in " + torch.cuda.get_device_name(0))  
        else:
            print("Training in CPU")

        if self.config.save_weights:
            if self.use_wandb:
                path_save_weights = self.config.root_path + wandb.run.id + "_" + self.config.save_path
            else:
                path_save_weights = self.config.root_path + self.config.save_path
            try:
                os.mkdir(path_save_weights)
            except OSError:
                pass

        model = (self.e1, self.e2, self.d1, self.d2, self.e_shared, self.d_shared, self.c_dann, self.discriminator1, self.denoiser)

        train_loader_faces, test_loader_faces, train_loader_cartoons, test_loader_cartoons = get_datasets(self.config.root_path, self.config.dataset_path_faces, self.config.dataset_path_cartoons, self.config.batch_size)
        optimizers = init_optimizers(model, self.config.learning_rate_opDisc, self.config.learning_rate_opTotal, self.config.learning_rate_denoiser, self.config.learning_rate_opCdann)

        criterion_bc = nn.BCEWithLogitsLoss()
        criterion_bc.to(self.device)

        images_faces_to_test = get_test_images(self.segmentation, self.config.batch_size, self.config.root_path + self.config.dataset_path_test_faces, self.config.root_path + self.config.dataset_path_segmented_faces)

        for epoch in tqdm(range(self.config.num_epochs)):
            loss_rec1, loss_rec2, loss_dann, loss_sem1, loss_sem2, loss_disc1, loss_gen1, loss_total, loss_denoiser, loss_teach = self.train_step(train_loader_faces, train_loader_cartoons, optimizers, criterion_bc)

            metrics_log = {"train_epoch": epoch+1,
                        "loss_rec1": loss_rec1.item(),
                        "loss_rec2": loss_rec2.item(),
                        "loss_dann": loss_dann.item(),
                        "loss_semantic12": loss_sem1.item(),
                        "loss_semantic21": loss_sem2.item(),
                        "loss_disc1": loss_disc1.item(),
                        "loss_gen1": loss_gen1.item(),
                        "loss_teach": loss_teach.item(),
                        "loss_total": loss_total.item()}

            if self.config.save_weights and ((epoch+1) % int(self.config.num_epochs/self.config.num_backups)) == 0:
                path_save_epoch = path_save_weights + 'epoch_{}'.format(epoch+1)
                try:
                    os.mkdir(path_save_epoch)
                except OSError:
                    pass
                save_weights(model, path_save_epoch, self.use_wandb)
                loss_test, fid_test, mmd_test, wandb_scatter_plot_1_fe_ce, wandb_scatter_plot_2_fe_cce, img_scatter_plot_1_fe_ce, img_scatter_plot_2_fe_cce = self.get_loss_test_set(test_loader_faces, test_loader_cartoons, criterion_bc)
                generated_images = test_image(model, self.device, images_faces_to_test)
                
                metrics_log["loss_total_test"] = loss_test
                metrics_log["fid_last_batch_test"] = fid_test
                metrics_log["mmd_last_batch_test"] = mmd_test
                metrics_log["Generated images"] = [wandb.Image(img) for img in generated_images]
                metrics_log['t-SNE evaluation plot 1 - FE and CE'] = wandb_scatter_plot_1_fe_ce
                metrics_log['t-SNE evaluation plot 2 - FE and CCE'] = wandb_scatter_plot_2_fe_cce
                
                if self.use_wandb: 
                    metrics_log["t-SNE evaluation images"] = [wandb.Image(img) for img in [img_scatter_plot_1_fe_ce, img_scatter_plot_2_fe_cce]]

            if self.use_wandb:
                wandb.log(metrics_log)


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
